import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run IREE Benchmark.")
    parser.add_argument("--parameters", required=True, help="Path to IRPA file")
    parser.add_argument("--vmfb", default=None, help="Path to VMFB file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--bs-prefill", default="1,2,4,8", help="Prefill batch sizes (default: 1,2,4,8)")
    parser.add_argument("--bs-decode", default="4,8,16,32,64", help="Decode batch sizes (default: 4,8,16,32,64)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "output_artifacts"
    vmfb = args.vmfb or str(output_dir / "output.vmfb")
    benchmark_dir = output_dir / "benchmark_module"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    irpa_path = args.parameters
    model = args.model
    print(f"Model: {model}")

    # LLaMA 8B FP8
    if model == "llama-8B-FP8":
        benchmarks = [
            ("prefill_bs4", "4x128xi64 4xi64 4x4xi64 261x2097152xf8E4M3FNUZ", 128),
            ("decode_bs4", "4x1xi64 4xi64 4xi64 4x5xi64 261x2097152xf8E4M3FNUZ", 128),
            ("prefill_bs4", "4x2048xi64 4xi64 4x64xi64 261x2097152xf8E4M3FNUZ", 2048),
            ("decode_bs4", "4x1xi64 4xi64 4xi64 4x65xi64 261x2097152xf8E4M3FNUZ", 2048),
        ]
        for func, inputs, isl in benchmarks:
            out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
            print(f"Running {model} {func} ISL: {isl}")
            run_cmd([
                "iree-benchmark-module",
                "--hip_use_streams=true",
                f"--module={vmfb}",
                f"--parameters=model={irpa_path}",
                "--device=hip",
                f"--function={func}",
                *[f"--input={i}" for i in inputs.split()],
                "--benchmark_repetitions=3",
                "--benchmark_out_format=json",
                f"--benchmark_out={out_file}",
            ])

    # LLaMA 8B FP16
    elif model == "llama-8B-FP16":
        benchmarks = [
            ("prefill_bs4", [
                "@/shark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/tokens.npy",
                "@/shark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_lens.npy",
                "@/shark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_block_ids.npy",
                "@/shark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/cs_f16.npy"
            ], 128),
            ("decode_bs4", [
                "@/shark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/next_tokens.npy",
                "@/shark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_lens.npy",
                "@/shark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/start_positions.npy",
                "@/shark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_block_ids.npy",
                "@/shark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/cs_f16.npy"
            ], 128),
            ("prefill_bs4", [
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/tokens.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/seq_lens.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/seq_block_ids.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/cs_f16.npy"
            ], 2048),
            ("decode_bs4", [
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/next_tokens.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/seq_lens.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/start_positions.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/seq_block_ids.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/cs_f16.npy"
            ], 2048),
        ]
        for func, inputs, isl in benchmarks:
            out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
            print(f"Running {model} {func} ISL: {isl}")
            run_cmd([
                "iree-benchmark-module",
                "--hip_use_streams=true",
                f"--module={vmfb}",
                f"--parameters=model={irpa_path}",
                "--device=hip",
                f"--function={func}",
                *[f"--input={i}" for i in inputs],
                "--benchmark_repetitions=3",
                "--benchmark_out_format=json",
                f"--benchmark_out={out_file}",
            ])

    # LLaMA 70B FP16
    elif model == "llama-70B-FP16":
        benchmarks = [
            ("prefill_bs4", ["4x128xsi64", "4xsi64", "4x4xsi64", "261x5242880xf16"], 128),
            ("decode_bs4", ["4x1xsi64", "4xsi64", "4xsi64", "4x5xsi64", "261x5242880xf16"], 128),
            ("prefill_bs4", ["4x2048xsi64", "4xsi64", "4x64xsi64", "513x5242880xf16"], 2048),
            ("decode_bs4", ["4x1xsi64", "4xsi64", "4xsi64", "4x65xsi64", "513x5242880xf16"], 2048),
        ]
        for func, inputs, isl in benchmarks:
            out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
            print(f"Running {model} {func} ISL: {isl}")
            run_cmd([
                "iree-benchmark-module",
                "--hip_use_streams=true",
                f"--module={vmfb}",
                f"--parameters=model={irpa_path}",
                "--device=hip",
                f"--function={func}",
                *[f"--input={i}" for i in inputs],
                "--benchmark_repetitions=3",
                "--benchmark_out_format=json",
                f"--benchmark_out={out_file}",
            ])

    # Mistral Nemo Instruct FP8
    elif model == "mistral-nemo-instruct-fp8":
        benchmarks = [
            ("prefill_bs1", "1x2048xsi64 1xsi64 1x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("prefill_bs2", "2x2048xsi64 2xsi64 2x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("prefill_bs4", "4x2048xsi64 4xsi64 4x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("prefill_bs8", "8x2048xsi64 8xsi64 8x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("decode_bs8", "8x1xsi64 8xsi64 8xsi64 8x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("decode_bs16", "16x1xsi64 16xsi64 16xsi64 16x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("decode_bs32", "32x1xsi64 32xsi64 32xsi64 32x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
            ("decode_bs64", "64x1xsi64 64xsi64 64xsi64 64x32xsi64 2048x2621440xf8E4M3FNUZ", 2048),
        ]
        for func, inputs, isl in benchmarks:
            out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
            print(f"Running {model} {func} ISL: {isl}")
            run_cmd([
                "iree-benchmark-module",
                "--device=hip",
                "--device_allocator=caching",
                f"--module={vmfb}",
                f"--parameters=model={irpa_path}",
                f"--function={func}",
                *[f"--input={i}" for i in inputs.split()],
                "--benchmark_repetitions=5",
                "--benchmark_out_format=json",
                f"--benchmark_out={out_file}",
            ])

    else:
        print(f"{model} test not implemented")
        raise ValueError(f"Unsupported model: {model}")

if __name__ == "__main__":
    main()
