r"""
Generate BOO dispatches for the specified kernels and tune them. Kernel
configurations follow the MIOpen driver format.

Example usage:
shark-boo-tuner \
  convbfp16 -n 6 -c 112 -H 1 -W 1 -k 448 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 \
  --output-td-spec tuning_spec.mlir --num-candidates 5000 --devices='hip://0'
"""

import argparse
import shlex
from pathlib import Path
import os
import tempfile
import subprocess
import shutil
import traceback

from model_tuner.model_tuner import main as tuner_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--commands-file",
        type=str,
        help="""
        Read commands from a file. Each line of the file should be an MIOpen
        command to tune. Any additional command-line arguments will be appended
        to each command from the file.
        """,
    )
    parser.add_argument("--output-td-spec", type=Path, default="tuning-spec.mlir")
    parser.add_argument("--starter-td-spec", type=Path)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--devices", type=str, default="hip://0")
    parser.add_argument(
        "--tmp-dir", type=str, default="", help="Directory to save temporary files."
    )
    args, extra_cli_args = parser.parse_known_args()

    # These imports are slow due to a pytorch dependency. Keeping them local
    # helps get fast '--help' output.
    from iree.turbine.kernel.boo import runtime as boo_runtime
    from iree.turbine.kernel.boo.driver.launch import get_launchable
    from iree.turbine.kernel.boo.op_exports.conv import ConvParser as mio

    # Parse MIOpen commands from a file if one was specified. Otherwise, just do
    # a single run with arguments from the command line.
    mio_file_args: list[list[str]] = [[]]
    if args.commands_file:
        with open(args.commands_file) as f:
            mio_file_args = [
                shlex.split(s) for s in f.readlines() if not s.startswith("#")
            ]

    starter_td_spec: Path | None = args.starter_td_spec
    for idx, file_args in enumerate(mio_file_args):
        cli_args = file_args + extra_cli_args
        print(f">>> ({idx+1}/{len(mio_file_args)}) {shlex.join(cli_args)}")
        parser = mio.get_miopen_parser()
        parser.add_argument("-t")
        parser.add_argument("--iter")
        sig = mio.get_signature(parser.parse_args(cli_args))
        if args.tmp_dir:
            tmp_dir = Path(args.tmp_dir)
            # Make sure directory is empty.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.mkdir(tmp_dir)
        else:
            tmp_dir = Path(tempfile.mkdtemp(dir=".", prefix="boo-tuner-"))
        boo_cache_dir = tmp_dir / "boo_cache"

        # Run BOO compilation an extract source IR.
        with boo_runtime.use_cache_dir(boo_cache_dir):
            get_launchable(sig)(*sig.get_sample_args(device="cuda", seed=123))
        [op_cache_dir] = os.listdir(boo_cache_dir)
        [compile_command_path] = [
            f
            for f in os.listdir(boo_cache_dir / op_cache_dir)
            if f.startswith("compile_command")
        ]
        with open(boo_cache_dir / op_cache_dir / compile_command_path) as f:
            compile_command = f.read().strip()

        # Re-compile to dump dispatches.
        bench_dir = tmp_dir / "bench"
        compile_args = shlex.split(compile_command) + [
            "--iree-codegen-tuning-spec-path=",  # Disable tuning spec flag as it prevents 'iree-config-add-tuner-attributes' from working.
            "--iree-config-add-tuner-attributes",
            "--iree-hal-dump-executable-benchmarks-to",
            str(bench_dir),
            "-o",
            os.devnull,
        ]
        print(f"> {shlex.join(compile_args)}")
        subprocess.run(compile_args)

        # Retain temporary directory if it was specified.
        should_cleanup_tmp_dir = not args.tmp_dir
        for bench_file in os.listdir(bench_dir):
            bench_path = bench_dir / bench_file

            starter_td_spec_args = (
                (f"--starter-td-spec={starter_td_spec}",)
                if starter_td_spec is not None
                else ()
            )
            # Run tuner.
            tuner_args = [
                "model-unused",
                str(bench_path),
                *starter_td_spec_args,
                f"--output-td-spec={args.output_td_spec}",
                f"--devices={args.devices}",
                "--model-tuner-num-dispatch-candidates=100",
                f"--num-candidates={args.num_candidates}",
                "--codegen-pipeline=llvmgpu_tile_and_fuse",
                "--stop-after=benchmark-dispatches",
            ]
            print(f"> {shlex.join(tuner_args)}")
            try:
                tuner_main(tuner_args)
                starter_td_spec = args.output_td_spec
            except Exception as err:
                # Print error and continue.
                traceback.print_exception(err)
                # Retain temporary directory for debugging.
                should_cleanup_tmp_dir = False
        if should_cleanup_tmp_dir:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
