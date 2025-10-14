# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Add a standalone benchmark driver executable, link dependencies and
# place it in the 'build/bin/benchmarks' sub-directory.
add_executable(fusilli_benchmark_driver driver.cpp)
target_link_libraries(fusilli_benchmark_driver PRIVATE
  libfusilli
  libutils
  CLI11::CLI11
)
set_target_properties(
  fusilli_benchmark_driver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

# Add some benchmark configurations for CI coverage.
add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nchw_fp32
  ARGS
    --iter 10 conv -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
  SRCS driver.cpp
  DEPS
    libfusilli
    libutils
    CLI11::CLI11
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nhwc_fp16
  ARGS
    --iter 10 conv --fp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 2 -j 2 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2
  SRCS driver.cpp
  DEPS
    libfusilli
    libutils
    CLI11::CLI11
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv3d_ndhwc_bf16
  ARGS
    --iter 10 conv --bf16 -n 16 -c 288 --in_d 2 -H 48 -W 32 -k 288 --fil_d 2 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --in_layout "NDHWC" --out_layout "NDHWC" --fil_layout "NDHWC" --spatial_dim 3
  SRCS driver.cpp
  DEPS
    libfusilli
    libutils
    CLI11::CLI11
)
