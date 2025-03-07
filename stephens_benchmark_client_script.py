import time
import random
import string
import argparse
import requests


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Benchmark Shortfin API.")
parser.add_argument(
    "--batch_size", type=int, default=4, help="Number of requests sent in parallel"
)
parser.add_argument(
    "--n_beams",
    type=int,
    default=8,
    help="number of sequences generated for each prompt",
)
parser.add_argument(
    "--base_url",
    type=str,
    default="http://localhost:32567",
    help="Base URL that the Shortfin server is running at.",
)

args = parser.parse_args()

# Completion API
# API definition https://platform.openai.com/docs/guides/completions https://platform.openai.com/docs/api-reference/completions/create

# Generate a token prompt
def generate_token_sequences(batch_size, length):
    """
    Generate an array of token sequences.

    Args:
        batch_size (int): Number of sequences to generate.
        length (int): Number of tokens in each sequence.

    Returns:
        list of list: A batch of token sequences.
    """

    def random_token():
        """Generate a random token (a string of random characters)."""
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=random.randint(1, 5))
        )

    return [" ".join(random_token() for _ in range(length)) for _ in range(batch_size)]


# Function to call OpenAI completion API
def generate_completion(base_url: str, prompt, sampling_params):
    # Generate outputs for the input prompts
    generation_endpoint = f"{base_url}/generate"
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "return_metrics": True,
    }
    print("Payload", len(prompt))
    return requests.post(generation_endpoint, json=payload)


# Test parameters
batch_size = args.batch_size  # Get batch size from argument
length = 1024  # Number of tokens in each prompt
max_tokens = 64  # generate max_tokens
num_sequences_generated = args.n_beams  # number of sequence generated for each prompt

num_tests = 3  # Total number of times to measure throughput

# Generate prompts for test
prompt = generate_token_sequences(batch_size, length)

# Define sampling parameters
sampling_params = {
    "n_beams": num_sequences_generated,
    "max_completion_tokens": max_tokens,
    "temperature": 0.8,
    # "top_p": 0.95,
}
# sampling_params = SamplingParams(n=num_sequences_generated, max_tokens= max_tokens, temperature=0.8, top_p=0.95)

# Initialize the LLM engine
# llm = LLM(model="../mistralai/Mistral-Nemo-Instruct-2407")

# Profile end-to-end latency
# warm_up
for i in range(2):
    completion = generate_completion(args.base_url, prompt, sampling_params)

latencies = []
# test
for _ in range(num_tests):
    request_start = time.perf_counter()
    completion = generate_completion(args.base_url, prompt, sampling_params)
    request_end = time.perf_counter()
    latencies.append(request_end - request_start)
    print(latencies)

# Metrics
avg_latency = sum(latencies) / len(latencies)
throughput = batch_size / avg_latency  # Requests per second

# Print results
print(f"\nPerformance Metrics:")
print(f"Batch size: {batch_size}, boN: {num_sequences_generated}")
print(f"Average latency per request: {avg_latency:.2f} s")
print(f"Throughput: {throughput:.2f} r/s")
print("\n" * 2)
