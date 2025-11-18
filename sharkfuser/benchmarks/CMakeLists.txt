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

# Test benchmark runner
if(FUSILLI_SYSTEMS_AMDGPU)
  add_test(
    NAME fusilli_benchmark_runner_tests
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark_runner.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.py
            $<TARGET_FILE:fusilli_benchmark_driver>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(fusilli_benchmark_runner_tests PROPERTIES
    LABELS "fusilli_benchmark_runner_tests"
    TIMEOUT 300
  )
endif()

# Add some benchmark configurations for CI coverage.
# Forward convolution benchmarks (mode=1)
add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nchw_fp32
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nhwc_fp16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 --fp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 2 -j 2 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv3d_ndhwc_bf16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 --bf16 -n 16 -c 288 --in_d 2 -H 48 -W 32 -k 288 --fil_d 2 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --in_layout "NDHWC" --out_layout "NDHWC" --fil_layout "NDHWC" --spatial_dim 3
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nchw_fp32_bias
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2 --bias
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv3d_ndhwc_bf16_bias
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 --bf16 -n 16 -c 288 --in_d 2 -H 48 -W 32 -k 288 --fil_d 2 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --in_layout "NDHWC" --out_layout "NDHWC" --fil_layout "NDHWC" --spatial_dim 3 --bias
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_nhwc_fp16_bias
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 1 --fp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2 --bias
)

# Weight Gradient benchmarks (mode=4)
add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_wgrad_nchw_fp32
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 4 -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_wgrad_nhwc_fp16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 4 --fp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_wgrad_nchw_bf16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 4 --bf16 -n 64 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_wgrad_nhwc_bf16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 4 --bf16 -n 32 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2
)

# Data Gradient benchmarks (mode=2)
add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_dgrad_nchw_fp32
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 2 -n 100 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_dgrad_nhwc_fp16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 2 --fp16 -n 16 -c 3 -H 48 -W 32 -k 48 -y 3 -x 3 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 --in_layout "NHWC" --out_layout "NHWC" --fil_layout "NHWC" --spatial_dim 2
)

add_fusilli_benchmark(
  NAME fusilli_benchmark_conv2d_dgrad_nchw_bf16
  DRIVER fusilli_benchmark_driver
  ARGS
    --iter 10 conv -F 2 --bf16 -n 64 -c 3 -H 32 -W 32 -k 32 -y 3 -x 3 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 --in_layout "NCHW" --fil_layout "NCHW" --out_layout "NCHW" --spatial_dim 2
)
