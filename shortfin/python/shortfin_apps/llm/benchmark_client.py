import asyncio
import time
import requests
import uuid
from typing import Dict, Any, List, Tuple
import numpy as np
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import argparse


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        max_workers: int = 128,
        stream: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stream = stream

    async def generate(
        self,
        text: str,
        sampling_params: Dict[str, Any] = None,
        save_output: bool = False,
    ) -> Dict[str, Any]:
        """Send a generation request to the LLM server, and return the results."""
        data = {
            "text": text,
            "sampling_params": sampling_params or {},
            # "rid": uuid.uuid4().hex,
            "return_logprob": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": self.stream,
        }

        headers = {"Content-Type": "application/json"}

        results = {}

        def process_stream():
            start_time = time.perf_counter()
            token_times = []
            generated_text = []
            if self.stream:
                try:
                    print(f"Sending request to {self.base_url}/generate")
                    with requests.post(
                        f"{self.base_url}/generate",
                        headers=headers,
                        json=data,
                        stream=self.stream,
                        timeout=1000,
                    ) as response:
                        response.raise_for_status()

                        # Process the response as it arrives
                        for line in response.iter_lines():
                            if line:
                                line_text = line.decode("utf-8")
                                if line_text.startswith("data"):
                                    token_time = time.perf_counter()
                                    token_times.append(token_time)
                                    if save_output:
                                        generated_text.append(
                                            line_text.split(": ", 1)[1]
                                        )
                    print(f"Received response from {self.base_url}/generate")
                except Exception as e:
                    print(f"Error in process_stream: {e}")
                    # Return empty results in case of error
                    return start_time, [], []
            else:
                response = requests.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=data,
                    timeout=1000,
                )
                if response.status_code != 200:
                    print(f"Error in response: {response.status_code} {response.text}")
                    return start_time, [], []
                else:
                    # print(f"Received response from {self.base_url}/generate")
                    # print(response.json())
                    token_times.append(time.perf_counter())
                    response.raise_for_status()
                    # generated_text = response.json()["text"]

            return start_time, token_times, generated_text

        # Run the processing function in the thread pool with our dedicated executor
        (
            start_time,
            token_times,
            generated_text,
        ) = await asyncio.get_event_loop().run_in_executor(
            self.executor, process_stream
        )

        # Handle case where no tokens were generated
        # if not token_times:
        #     results["metrics"] = {
        #         "start_time": start_time,
        #         "end_time": end_time,
        #         "time_to_first_token": 0,
        #         "token_generation_times": [],
        #         "num_tokens": 0,
        #         "generated_text": "",
        #         "error": "No tokens generated",
        #     }
        #     return results

        time_to_first_token = token_times[0] - start_time if token_times else 0
        num_tokens = len(token_times)
        end_time = token_times[-1]

        results["metrics"] = {
            "start_time": start_time,
            "end_time": end_time,
            "num_tokens": num_tokens,
            "generated_text": "".join(generated_text) if save_output else "",
            "is_streaming": self.stream,
        }

        if self.stream:
            results["metrics"]["time_to_first_token"] = time_to_first_token
            results["metrics"]["token_generation_times"] = token_times

        return results

    def __del__(self):
        # Ensure executor is shut down when client is destroyed
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


def calculate_metrics(
    results: List[Dict[str, Any]], start_time: float
) -> Dict[str, Any]:
    """Calculate benchmark metrics from results."""
    num_concurrent_requests = len(results)
    is_streaming = results[0]["metrics"]["is_streaming"]

    # Extract raw metrics
    if is_streaming:
        token_generation_times = [
            result["metrics"]["token_generation_times"] for result in results
        ]
        time_to_first_token = [
            result["metrics"]["time_to_first_token"] for result in results
        ]
        num_generated_tokens = [result["metrics"]["num_tokens"] for result in results]

        # Calculate token-level metrics
        flattened_token_times = sorted(
            [item for sublist in token_generation_times for item in sublist]
        )

        TPS_times = [
            flattened_token_times[i] - flattened_token_times[i - 1]
            for i in range(1, len(flattened_token_times))
        ]
        TPOT_times = [
            [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            for token_times in token_generation_times
        ]
        TPOT_times = [item for sublist in TPOT_times for item in sublist]

    start_times = [result["metrics"]["start_time"] - start_time for result in results]
    end_times = [result["metrics"]["end_time"] - start_time for result in results]
    time_per_request = [
        end_times[i] - start_times[i] for i in range(num_concurrent_requests)
    ]

    E2E_latency = end_times[-1] - start_times[0]

    if is_streaming:
        return {
            "E2E_latency": E2E_latency,
            "time_to_first_token": time_to_first_token,
            "TPOT_times": TPOT_times,
            "TPS_times": TPS_times,
            "start_times": start_times,
            "end_times": end_times,
            "time_per_request": time_per_request,
            "num_generated_tokens": num_generated_tokens,
        }
    else:
        return {
            "E2E_latency": E2E_latency,
            "start_times": start_times,
            "end_times": end_times,
            "time_per_request": time_per_request,
        }


def print_benchmark_results(metrics: Dict[str, Any], config: Dict[str, Any]):
    num_concurrent_requests = len(metrics["start_times"])

    print("\nBenchmark Configuration:")
    print(f"Input token length: {config['input_token_length']}")
    print(f"Output token length: {config['output_token_length']}")
    print(f"Number of concurrent requests: {num_concurrent_requests}")
    print(f"Token selection strategy: {config['token_selection_strategy']}")

    print("\nPerformance Metrics:")
    print(f"E2E latency: {metrics['E2E_latency']:.2f} seconds")
    print(f"Requests per second: {num_concurrent_requests/metrics['E2E_latency']:.2f}")
    print(f"Average latency: {np.mean(metrics['time_per_request']):.2f} seconds")

    print("\nDetailed Metrics:")
    metric_names = {
        "time_to_first_token": "Time to first token",
        "TPOT_times": "Time per output token",
        "TPS_times": "Tokens per second",
        "start_times": "Request processing start time",
        "end_times": "Request processing end time",
        "time_per_request": "Time per request",
        "num_generated_tokens": "Number of generated tokens",
    }

    for metric_key, metric_name in metric_names.items():
        values = metrics.get(metric_key, [])
        if values:
            print(
                f"{metric_name}: Mean: {np.mean(values):.4f}s, SD: {np.std(values):.4f}s, "
                f"Median: {np.median(values):.4f}s, Min: {np.min(values):.4f}s, "
                f"Max: {np.max(values):.4f}s"
            )


def get_csv_results(metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Get results in a format suitable for CSV export."""
    num_concurrent_requests = len(metrics["start_times"])

    csv_results = {
        "E2E_latency": metrics["E2E_latency"],
        "token_selection_strategy": config["token_selection_strategy"],
        "input_token_length": config["input_token_length"],
        "output_token_length": config["output_token_length"],
        "num_concurrent_requests": num_concurrent_requests,
        "total_time": metrics["E2E_latency"],
        "requests_per_second": num_concurrent_requests / metrics["E2E_latency"],
        "avg_latency": np.mean(metrics["time_per_request"]),
        "Time_per_request_median": np.median(metrics["time_per_request"]),
    }

    if metrics.get("time_to_first_token"):
        csv_results["TTFT_median"] = np.median(metrics["time_to_first_token"])
        csv_results["TPOT_median"] = np.median(metrics["TPOT_times"])
        csv_results["TPS_median"] = np.median(metrics["TPS_times"])
    return csv_results


async def run_benchmark(
    input_token_length: int = 100,
    output_token_length: int = 50,
    num_concurrent_requests: int = 64,
    token_selection_strategy: str = "multi_greedy",
    endpoint: str = "http://localhost:8080",
    streaming = False,
    multi_hypothesis = False,
    best_of_n = 8,
    top_p = 0.95
):
    """Execute the benchmark and return raw data."""
    client = LLMClient(base_url=endpoint, stream=streaming)

    prompt = " ".join(["one" for _ in range(input_token_length)])
    config = {
        "input_token_length": input_token_length,
        "output_token_length": output_token_length,
        "token_selection_strategy": token_selection_strategy,
    }

    params = {
            "max_completion_tokens": output_token_length,
            "token_selection_strategy": token_selection_strategy,
            "num_beams": 8,
    }

    if multi_hypothesis:
        params["b_of_n"] = best_of_n
        params["top_p"] = top_p

    # Create tasks
    tasks = []
    # Run benchmark
    start_time = time.perf_counter()
    for _ in range(num_concurrent_requests):
        tasks.append(
            client.generate(
                text=prompt,
                sampling_params=params,
                save_output=False,
            )
        )

    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()

    # Return raw data for processing
    return {
        "raw_results": results,
        "start_time": start_time,
        "end_time": end_time,
        "config": config,
        "num_concurrent_requests": num_concurrent_requests,
    }


def compute_benchmark_results(benchmark_data):
    """Compute benchmark results from raw data."""
    raw_results = benchmark_data["raw_results"]
    start_time = benchmark_data["start_time"]
    end_time = benchmark_data["end_time"]
    config = benchmark_data["config"]
    num_concurrent_requests = benchmark_data["num_concurrent_requests"]

    # Calculate metrics
    metrics = calculate_metrics(raw_results, start_time)

    # Print results
    print_benchmark_results(metrics, config)

    # Get CSV results
    csv_results = get_csv_results(metrics, config)

    return csv_results


async def continuous_load_test(
    input_token_length: int,
    output_token_length: int,
    token_selection_strategy: str,
    endpoint: str,
    duration: int = 60,  # Run for 60 seconds by default
    streaming = False,
    multi_hypothesis = False,
    best_of_n = 8,
    top_p = 0.95
) -> Dict[str, Any]:
    """Run a continuous load test with a single client sending requests continuously."""
    client = LLMClient(base_url=endpoint, stream=streaming)
    prompt = " ".join(["one" for _ in range(input_token_length)])

    start_time = time.perf_counter()
    end_time = start_time + duration
    request_times = []
    num_requests = 0

    params = {
            "max_completion_tokens": output_token_length,
            "token_selection_strategy": token_selection_strategy,
            "num_beams": 8,
    }

    if multi_hypothesis:
        params["b_of_n"] = best_of_n
        params["top_p"] = top_p


    while time.perf_counter() < end_time:
        try:
            request_start = time.perf_counter()
            await client.generate(
                text=prompt,
                sampling_params=params,
                save_output=False,
            )
            request_end = time.perf_counter()
            request_times.append(request_end - request_start)
            num_requests += 1
        except Exception as e:
            print(f"Error in request: {e}")
            continue

    actual_duration = time.perf_counter() - start_time
    return {
        "num_requests": num_requests,
        "duration": actual_duration,
        "avg_latency": np.mean(request_times) if request_times else 0,
        "min_latency": np.min(request_times) if request_times else 0,
        "max_latency": np.max(request_times) if request_times else 0,
    }


async def calculate_throughput(
    input_token_length: int,
    output_token_length: int,
    num_concurrent_requests: int,
    token_selection_strategy: str,
    endpoint: str,
    duration: int = 60,  # Run for 60 seconds by default
    streaming = False,
    multi_hypothesis = False,
    best_of_n = 8,
    top_p = 0.95
):
    """Calculate throughput by running continuous load tests with multiple concurrent clients."""
    print(
        f"\nCalculating throughput with {num_concurrent_requests} concurrent clients for {duration} seconds..."
    )

    # Create tasks for each concurrent client
    tasks = []
    for _ in range(num_concurrent_requests):
        tasks.append(
            continuous_load_test(
                input_token_length=input_token_length,
                output_token_length=output_token_length,
                token_selection_strategy=token_selection_strategy,
                endpoint=endpoint,
                duration=duration,
                streaming=streaming,
                multi_hypothesis=multi_hypothesis,
                best_of_n=best_of_n,
                top_p=top_p
            )
        )

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Aggregate results
    total_requests = sum(r["num_requests"] for r in results)
    avg_duration = np.mean([r["duration"] for r in results])
    throughput = total_requests / avg_duration

    # Calculate aggregate latency statistics
    all_latencies = []
    for r in results:
        if r["avg_latency"] > 0:  # Only include clients that successfully made requests
            all_latencies.extend([r["min_latency"], r["avg_latency"], r["max_latency"]])

    if all_latencies:
        avg_latency = np.mean(all_latencies)
        min_latency = np.min(all_latencies)
        max_latency = np.max(all_latencies)
    else:
        avg_latency = min_latency = max_latency = 0

    print(f"\nThroughput Results:")
    print(f"Number of concurrent clients: {num_concurrent_requests}")
    print(f"Total requests completed: {total_requests}")
    print(f"Test duration: {avg_duration:.2f} seconds")
    print(f"Average latency: {avg_latency:.4f}s")
    print(f"Min latency: {min_latency:.4f}s")
    print(f"Max latency: {max_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} requests/second")

    return {
        "num_concurrent_requests": num_concurrent_requests,
        "total_requests": total_requests,
        "duration": avg_duration,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "throughput": throughput,
    }


async def run_all_benchmarks(
    input_token_lengths: List[int] = [1024],
    output_token_lengths: List[int] = [64],
    min_concurrent_requests: int = 10,
    max_concurrent_requests: int = 200,
    token_selection_strategy: str = "greedy",
    target_latency: float = 4.2,
    endpoint: str = "http://localhost:8080",
    num_throughput_runs: int = 20,
    results_dir: str = "results",
    multi_hypothesis = False,
    streaming = False,
    best_of_n = 8,
    top_p = 0.95
):
    all_results = []
    throughput_results = []

    for input_token_length in input_token_lengths:
        for output_token_length in output_token_lengths:
            print(
                f"\n\nRunning benchmarks with input_token_length = {input_token_length}, output_token_length = {output_token_length}"
            )

            # Start with minimum concurrent requests
            current_requests = min_concurrent_requests
            optimal_requests = None

            if multi_hypothesis:
                token_selection_strategy = "multi_greedy"

            while current_requests <= max_concurrent_requests:
                print(f"Testing with {current_requests} concurrent requests")
                benchmark_data = await run_benchmark(
                    input_token_length=input_token_length,
                    output_token_length=output_token_length,
                    num_concurrent_requests=current_requests,
                    token_selection_strategy=token_selection_strategy,
                    endpoint=endpoint,
                    streaming=streaming,
                    multi_hypothesis=multi_hypothesis,
                    best_of_n=best_of_n,
                    top_p=top_p
                )
                result = compute_benchmark_results(benchmark_data)
                all_results.append(result)

                # Check if we've reached our target latency
                if result["E2E_latency"] >= target_latency:
                    print(
                        f"Reached target latency of {target_latency}s with {current_requests} concurrent requests"
                    )
                    optimal_requests = current_requests
                    print(f"Optimal number of concurrent requests: {optimal_requests}")
                    break

                # Increase concurrent requests for next iteration
                # Using a step size that increases with the current request count
                step_size = max(1, current_requests // 10)
                current_requests += step_size

                # If we've exceeded max_concurrent_requests, break
                if current_requests > max_concurrent_requests:
                    print(
                        f"Reached maximum concurrent requests ({max_concurrent_requests}) without meeting target latency"
                    )
                    optimal_requests = max_concurrent_requests

            # Calculate throughput with the optimal number of concurrent requests
            if optimal_requests:
                throughput_result = await calculate_throughput(
                    input_token_length=input_token_length,
                    output_token_length=output_token_length,
                    num_concurrent_requests=optimal_requests,
                    token_selection_strategy=token_selection_strategy,
                    endpoint=endpoint,
                    duration=num_throughput_runs,
                )
                throughput_results.append(
                    {
                        "input_token_length": input_token_length,
                        "output_token_length": output_token_length,
                        "token_selection_strategy": token_selection_strategy,
                        **throughput_result,
                    }
                )

    # Create CSV file for benchmark results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    csv_filename = f"{results_dir}/benchmark_results_{token_selection_strategy}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nBenchmark results saved to {csv_filename}")

    # Create CSV file for throughput results
    if throughput_results:
        throughput_csv_filename = (
            f"{results_dir}/throughput_results_{token_selection_strategy}.csv"
        )

        with open(throughput_csv_filename, "w", newline="") as csvfile:
            fieldnames = throughput_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in throughput_results:
                writer.writerow(result)

        print(f"Throughput results saved to {throughput_csv_filename}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarking with configurable parameters"
    )

    # Add arguments
    parser.add_argument(
        "--input-token-lengths",
        type=int,
        nargs="+",
        default=[1024],
        help="List of input token lengths to test",
    )
    parser.add_argument(
        "--output-token-lengths",
        type=int,
        nargs="+",
        default=[64],
        help="List of output token lengths to test",
    )
    parser.add_argument(
        "--min-concurrent-requests",
        type=int,
        default=2,
        help="Minimum number of concurrent requests to start with",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=10,
        help="Maximum number of concurrent requests to test",
    )
    parser.add_argument(
        "--token-selection-strategy",
        type=str,
        default="greedy",
        choices=["greedy", "multi_greedy", "beam_search"],
        help="Token selection strategy to use",
    )
    parser.add_argument(
        "--target-latency",
        type=float,
        default=4.2,
        help="Target end-to-end latency in seconds",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="LLM server endpoint URL",
    )
    parser.add_argument(
        "--num-throughput-runs",
        type=int,
        default=20,
        help="Number of runs to calculate throughput",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--multi-hypothesis",
        action='store_true',
        help="Enable multi hypothesis",
    )
    parser.add_argument(
        "--stream",
        action='store_true',
        help="Enable streaming",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=8,
        help="Best of N (defaults to 8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top P value (defaults to 0.95)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run benchmarks with parsed arguments
    asyncio.run(
        run_all_benchmarks(
            input_token_lengths=args.input_token_lengths,
            output_token_lengths=args.output_token_lengths,
            min_concurrent_requests=args.min_concurrent_requests,
            max_concurrent_requests=args.max_concurrent_requests,
            token_selection_strategy=args.token_selection_strategy,
            target_latency=args.target_latency,
            endpoint=args.endpoint,
            num_throughput_runs=args.num_throughput_runs,
            results_dir=args.results_dir,
            streaming=args.stream,
            multi_hypothesis=args.multi_hypothesis,
            best_of_n=args.best_of_n,
            top_p = args.top_p
        )
    )
