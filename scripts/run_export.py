# run export
import argparse
import os
import subprocess
import sys
import time

def run_command(cmd, **kwargs):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Export IR with IREE")
    parser.add_argument("--irpa", default="/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa",
                        help="Path to IRPA file")
    parser.add_argument("--bs-prefill", default="1,2,4,8", help="Prefill batch sizes")
    parser.add_argument("--bs-decode", default="4,8,16,32,64", help="Decode batch sizes")
    parser.add_argument("--dtype", default="fp16", help="Data type (fp16/fp8/mistral_fp8)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for dumping artifacts")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    # Export Step
    print("Exporting IR ....")
    start = time.time()

    export_cmd = [
        sys.executable, "-m", "sharktank.examples.export_paged_llm_v1",
        f"--irpa-file={args.irpa}",
        f"--output-mlir={os.path.join(output_dir, 'output.mlir')}",
        f"--output-config={os.path.join(output_dir, 'config_attn.json')}",
        f"--bs-prefill={args.bs_prefill}",
        f"--bs-decode={args.bs_decode}"
    ]

    if args.dtype == "fp8":
        export_cmd += [
            "--attention-kernel", "sharktank",
            "--use-hf"
        ]
    elif args.dtype == "mistral_fp8":
        export_cmd += [
            "--use-hf",
            "--attention-kernel", "torch"
        ]

    export_cmd += [
            "--device-block-count", "4096",
            "--attention-dtype", "float16",
            "--activation-dtype", "float16",
            "--kv-cache-dtype", "float8_e4m3fnuz"
        ]

    run_command(export_cmd)
    print(f"Time taken for exporting: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    main()
