# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import subprocess
import sys

from pathlib import Path

verbose = None


def print_cmd(args: str | list[str]):
    if isinstance(args, str):
        cmd_str = args
    else:
        cmd_str = " ".join((f'"{a}"' for a in args))
    print(cmd_str)


def check_call(args: str | list[str], *argv, **kwargs):
    if verbose:
        print_cmd(args)
    subprocess.check_call(args, *argv, **kwargs)


def main():
    unspecified_arg_value = "__unspecified__"

    parser = argparse.ArgumentParser(description="Prepare CI environment.")
    parser.add_argument(
        "--torch-version",
        type=str,
        default=None,
        help="Version of PyTorch. If omitted will install the default from pytorch-*-requirements.txt.",
    )
    parser.add_argument(
        "--pytorch-rocm",
        action="store_true",
        default=False,
        help="Install PyTorch for ROCm instead of for CPU.",
    )
    parser.add_argument(
        "--python-venv-dir",
        type=Path,
        default=unspecified_arg_value,
        help="Python venv to create and activate.",
    )
    parser.add_argument(
        "--iree-unpinned",
        action="store_true",
        default=False,
        help="Install IREE unpinned/pinned version.",
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()
    src_dir = Path(
        os.environ.get("SHARK_AI_SRC_DIR", f"{Path(__file__).parent / '..' / '..'}")
    )

    global verbose
    verbose = args.verbose

    if str(args.python_venv_dir) != unspecified_arg_value:
        assert sys.platform != "win32", "TODO: implement venv activation on Windows"
        if not args.python_venv_dir.exists():
            check_call(
                f'"{sys.executable}" -m venv "{args.python_venv_dir}"', shell=True
            )
        argv_str = " ".join((f'"{a}"' for a in sys.argv))
        check_call(
            (
                f'. "{args.python_venv_dir}/bin/activate" && '
                f'"{sys.executable}" {argv_str} --python-venv-dir={unspecified_arg_value}'
            ),
            shell=True,
        )

    check_call(
        f"{sys.executable} -m pip install --no-compile --upgrade pip", shell=True
    )

    if args.torch_version is None:
        if args.pytorch_rocm:
            requirements_file = src_dir / "pytorch-rocm-requirements.txt"
        else:
            requirements_file = src_dir / "pytorch-cpu-requirements.txt"
        check_call(f"pip install --no-compile -r {requirements_file}", shell=True)
    else:
        if args.pytorch_rocm:
            rocm_version = (
                subprocess.check_output(
                    [
                        sys.executable,
                        f"{src_dir / 'build_tools' / 'torch_rocm_version_map.py'}",
                        f"{args.torch_version}",
                    ]
                )
                .decode()
                .strip()
            )
            pkg_args = (
                f"--index-url https://download.pytorch.org/whl/rocm{rocm_version} "
                f"torch=={args.torch_version}+rocm{rocm_version}"
            )
        else:
            pkg_args = (
                "--index-url https://download.pytorch.org/whl/cpu "
                f"torch=={args.torch_version}+cpu"
            )
        check_call(f"pip install --no-compile {pkg_args}", shell=True)

    if args.iree_unpinned:
        check_call(
            f"pip install -r {src_dir / 'requirements-iree-unpinned.txt'}", shell=True
        )
    else:
        check_call(
            f"pip install -r {src_dir / 'requirements-iree-pinned.txt'}", shell=True
        )

    test_requirements = src_dir / "sharktank" / "requirements-tests.txt"
    check_call(f"pip install --no-compile -r {test_requirements}", shell=True)

    check_call(f"pip install --no-compile -e {src_dir / 'sharktank'}", shell=True)


if __name__ == "__main__":
    main()
