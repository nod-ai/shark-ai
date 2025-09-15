# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Get the ROCm version corresponding to the torch version of PyTorch ROCm."
    )
    parser.add_argument("torch_version", type=str)
    args = parser.parse_args()
    map = {"2.6.0": "6.2.4"}
    print(map[args.torch_version])


if __name__ == "__main__":
    main()
