# Compile 
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
    parser = argparse.ArgumentParser(description="Compile IR with IREE")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for dumping artifacts")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    print(" Compiling IR ....")
    start = time.time()

    compile_cmd = [
        "iree-compile",
        os.path.join(output_dir, "output.mlir"),
        "--iree-hip-target=gfx942",
        "-o", os.path.join(output_dir, "output.vmfb"),
        "--iree-opt-level=O3",
        "--iree-hal-indirect-command-buffers=true",
        "--iree-stream-resource-memory-model=discrete",
        "--iree-hip-enable-tensor-ukernels",
        "--iree-hal-memoization=true",
        "--iree-codegen-enable-default-tuning-specs=true"
    ]

    compile_cmd += ["--iree-hal-target-device=hip"]
    run_command(compile_cmd)

    print(f"Time taken for compiling: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    main()
