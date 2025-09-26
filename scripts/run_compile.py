"""
Compiles The Exported MLIR from Sharktank To vmfb
"""

import argparse
import os
import subprocess
import sys
import time
import ast
from pathlib import Path


def run_command(cmd, **kwargs):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Compile IR with IREE")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory For Dumping Artifacts"
    )
    parser.add_argument(
        "--dtype", default="fp16", help="Data Type (fp16/fp8/mistral_fp8)"
    )
    parser.add_argument(
        "--iree-hip-target", default="gfx942", help="IREE HIP Target To Compile For. Default: gfx942."
    )
    parser.add_argument(
        "--extra-compile-flags-list",
        type=str,
        default="[]",
        help="Extra Flags To Pass As A List like '['--x', '--f'', '--g']' or '[]'",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    os.environ["DTYPE"] = "fp16"
    if args.dtype == "fp8":
        os.environ["ATTENTION_DTYPE"] = "float16"
        os.environ["ACTIVATION_DTYPE"] = "float16"
        os.environ["KV_CACHE_DTYPE"] = "float8_e4m3fnuz"

    OUTPUT_DIR = (
        Path(os.getcwd()) / "output_artifacts"
    )  # override the output dir path for CI

    print(" Compiling IR ....")

    compile_cmd = [
        "iree-compile",
        os.path.join(OUTPUT_DIR, "output.mlir"),
        f"--iree-hip-target={args.iree_hip_target}",
        "-o",
        os.path.join(OUTPUT_DIR, "output.vmfb"),
        "--iree-hal-target-device=hip",
        "--iree-opt-level=O3",
        "--iree-hal-indirect-command-buffers=true",
        "--iree-stream-resource-memory-model=discrete",
        "--iree-hip-enable-tensor-ukernels",
        "--iree-hal-memoization=true",
        "--iree-codegen-enable-default-tuning-specs=true",
        "--iree-stream-affinity-solver-max-iterations=1024",
    ]

    try:
        extra_flags = ast.literal_eval(args.extra_compile_flags_list)
        if not isinstance(extra_flags, list):
            raise ValueError(
                f"Expected a list for --extra-compile-flags-list: {args.extra_compile_flags_list}"
            )
    except Exception as e:
        raise ValueError(
            f"Invalid value for --extra-compile-flags-list: {args.extra_compile_flags_list}"
        ) from e

    if len(extra_flags) == 0:
        print("No Extra Compile Flag is Passed")
    else:
        print("Appending Extra Compile Flags...")
        compile_cmd += extra_flags

    print(f"Using Compile Command: {compile_cmd}")

    start = time.time()
    run_command(compile_cmd)
    print(f"Time taken for compiling: {int(time.time() - start)} seconds")


if __name__ == "__main__":
    main()
