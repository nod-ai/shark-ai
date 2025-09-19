'''
IREE Benchmark Test.

'''
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
    # parser.add_argument("--bs-prefill", default="1,2,4,8", help="Prefill batch sizes (default: 1,2,4,8)")
    # parser.add_argument("--bs-decode", default="4,8,16,32,64", help="Decode batch sizes (default: 4,8,16,32,64)")
    # parser.add_argument("--extra-flags", default=[], help="Add Extra Flags That Have To Be Passed For a Specific Model in test/configs")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="[]",
        help="Extra flags to pass as a Python-style list, e.g. '[\"--x\", \"--f\", \"--g\"]' or '[]'"
)

    #parser.add_argument("--benchmarks", required=True, help="(see format in ../tests/configs.py file):<benchmark_name>, [<comma seperated input values>], <ISL>")
    parser.add_argument("--benchmark_repetition", required=True, help="eg: 3 (see format in ../tests/configs.py file): ")
    # parser.add_argument(
    #     "--extra-export-flags-list",
    #     type=str,
    #     default="[]",
    #     help="Extra flags to pass as a Python-style list, e.g. '[\"--x\", \"--f\", \"--g\"]' or '[]'"
    # )


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

    benchmark


    for benchmark in benchmarks:
        func = benchmark['name']
        inputs = benchmark['inputs']
        isl = benchmark.get('seq_len') # Adjust based on your requirements
        out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
        benchmark_command = [
            "iree-benchmark-module",
            # "--device_allocator=caching",
            "--hip_use_streams=true",
            f"--module={vmfb}",
            f"--parameters=model={irpa_path}",
            "--device=hip",
            f"--function={func}",
            *[f"--input={i}" for i in inputs],
            f"--benchmark_repetitions={args.benchmark_repetition}",
            "--benchmark_out_format=json",
            f"--benchmark_out={out_file}",
        ]

        #benchmark_cmd = benchmark_command + extra_benchmark_flags

        run_cmd(benchmark_command)

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    main()
