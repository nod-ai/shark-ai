"""
IREE Benchmark Test.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import ast


def run_cmd(cmd: list[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run IREE Benchmark.")
    parser.add_argument("--parameters", required=True, help="Path to IRPA file")
    parser.add_argument("--vmfb", required=True, help="Path to VMFB file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="[]",
        help="Function Name, Inputs And The Input Sequence Length.",
    )

    parser.add_argument(
        "--benchmark_repetition",
        required=True,
        help="Number Of Repeatitions For Benchmarks",
    )
    parser.add_argument(
        "--extra-benchmark-flags-list",
        type=str,
        default="[]",
        help="Extra flags To Pass As A List Like ['--hip_use_streams=true']",
    )

    args = parser.parse_args()

    try:
        benchmarks = ast.literal_eval(args.benchmarks)
        if not isinstance(benchmarks, list):
            raise ValueError("Expected a list for --benchmarks")
    except Exception as e:
        raise ValueError(f"Invalid value for --benchmarks: {args.benchmarks}") from e

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "output_artifacts"
    vmfb = args.vmfb or str(output_dir / "output.vmfb")
    benchmark_dir = output_dir / "benchmark_module"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    irpa_path = args.parameters
    model = args.model
    print(f"Model: {model}")

    benchmark_command = ["iree-benchmark-module"]

    try:
        extra_flags = [
            flag.strip() for flag in ast.literal_eval(args.extra_benchmark_flags_list)
        ]
        if not isinstance(extra_flags, list):
            raise ValueError(
                f"Invalid value for --extra-benchmark-flags-list: {args.extra_benchmark_flags_list}"
            ) from e
    except Exception as e:
        raise ValueError(
            f"Invalid value for --extra-benchmark-flags-list: {args.extra_benchmark_flags_list}"
        ) from e

    if len(extra_flags) == 0:
        print("No Extra Benchmark Flag Passed.")

    else:
        print("Appending Extra Benchmark Flags...")
        print(extra_flags)
        benchmark_command += extra_flags

    for benchmark in benchmarks:
        func = benchmark["name"]
        inputs = benchmark["inputs"]
        isl = benchmark.get("seq_len")
        out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
        command = [
            f"--module={vmfb}",
            f"--parameters=model={irpa_path}",
            f"--function={func}",
            *[f"--input={i}" for i in inputs],
            f"--benchmark_repetitions={args.benchmark_repetition}",
            "--benchmark_out_format=json",
            f"--benchmark_out={out_file}",
        ]

        print(f"\n Using Benchmark Command: {benchmark_command + command}\n")
        run_cmd(benchmark_command + command)


if __name__ == "__main__":
    main()
