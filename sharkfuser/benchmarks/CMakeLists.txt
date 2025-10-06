# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_fusilli_benchmark(
  NAME fusilli_driver
  ARGS
    conv -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1
  SRCS driver.cpp
  DEPS
    libfusilli
    libutils
    CLI11::CLI11
)
