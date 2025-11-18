# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import safetensors


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Print names/keys and optionally metadata of tensors in a safetensors file."
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print also tensor metadata, shape and dtype.",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to safetensors file.",
    )
    args = parser.parse_args(args)

    with safetensors.safe_open(args.file_path, framework="pt") as f:
        keys = [key for key in f.keys()]
        for key in keys:
            if args.show_metadata:
                tensor = f.get_tensor(key)
                print(f"{key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
            else:
                print(key)


if __name__ == "__main__":
    main()
