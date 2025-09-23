"""
Exports The MLIR from Sharktank
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
        start = time.time()
        subprocess.run(cmd, check=True, **kwargs)
        print(f"Time taken for exporting: {int(time.time() - start)} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Export IR with IREE")
    parser.add_argument("--irpa", required=True, help="Path to IRPA file")
    parser.add_argument("--bs-prefill", default=4, help="Prefill batch sizes")
    parser.add_argument("--bs-decode", default=4, help="Decode batch sizes")
    parser.add_argument(
        "--dtype", default="fp16", help="Data type (fp16/fp8/mistral_fp8)"
    )
    parser.add_argument(
        "--device-block-count",
        default="4096",
        help="What Device Block Count To Be Used",
    )
    parser.add_argument(
        "--extra-export-flags-list",
        type=str,
        default="[]",
        help="Extra flags to pass as a list, e.g. '['--x', '--f'', '--g']' or '[]'",
    )
    parser.add_argument(
        "--output-dir",
        default="../shark-ai/output_artifacts/",
        help="Output directory for dumping artifacts",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir  # or os.path.join(script_dir, "output_artifacts/")
    os.makedirs(output_dir, exist_ok=True)

    OUTPUT_DIR = (
        Path(os.getcwd()) / "output_artifacts"
    )  # override the output dir path for CI

    os.environ["OUTPUT_DIR"] = args.output_dir
    os.environ["IRPA_PATH"] = args.irpa
    os.environ["DTYPE"] = "fp16"
    if args.dtype == "fp8":
        os.environ["ATTENTION_DTYPE"] = "float16"
        os.environ["ACTIVATION_DTYPE"] = "float16"
        os.environ["KV_CACHE_DTYPE"] = "float8_e4m3fnuz"

    ###   Starting Export
    print("Exporting IR ....")

    export_cmd = [
        sys.executable,
        "-m",
        "sharktank.examples.export_paged_llm_v1",
        f"--irpa-file={args.irpa}",
        f"--output-mlir={os.path.join(OUTPUT_DIR, 'output.mlir')}",
        f"--output-config={os.path.join(output_dir, 'config_attn.json')}",
        f"--bs-prefill={args.bs_prefill}",
        f"--bs-decode={args.bs_decode}",
        f"--device-block-count",
        f"{args.device_block_count}",
    ]

    try:
        extra_flags = [
            flag.strip() for flag in ast.literal_eval(args.extra_export_flags_list)
        ]
        if not isinstance(extra_flags, list):
            raise ValueError(
                f"Invalid value for --extra-export-flags-list: {args.extra_export_flags_list}"
            ) from e
    except Exception as e:
        raise ValueError(
            f"Invalid value for --extra-export-flags-list: {args.extra_export_flags_list}"
        ) from e

    if len(extra_flags) == 0:
        print("No Extra Export Flag Passed.")
    else:
        print("Appending Extra Export Flags...")
        print(extra_flags)
        export_cmd += extra_flags
        print("Command:", export_cmd)

    print("Using Export Command:")
    print(export_cmd)
    run_command(export_cmd)


if __name__ == "__main__":
    main()
