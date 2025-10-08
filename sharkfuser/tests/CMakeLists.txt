# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find lit program
fusilli_find_program(lit "Please install lit (e.g., pip install lit).")

# Find iree-opt program
fusilli_find_program(iree-opt "Please install iree-opt (e.g., pip install iree-base-compiler).")

# Find FileCheck program
fusilli_find_program(FileCheck "Please install FileCheck.")

# Enable HIP tests if AMD GPU is enabled (ensures GPU detected) and HIP is found
if(FUSILLI_SYSTEMS_AMDGPU)
  set(HIP_PLATFORM "amd")
  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "AMD GPU detected and HIP found: Enabling HIP tests")
    add_subdirectory(hip_tests)
  else()
    message(STATUS "HIP not found: Skipping HIP tests")
  endif()
endif()

add_fusilli_test(
  NAME fusilli_attribute_tests
  SRCS
    test_attributes.cpp
    test_tensor_attributes.cpp
    test_conv_attributes.cpp
    test_pointwise_attributes.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_fusilli_test(
  NAME fusilli_node_tests
  SRCS
    test_conv_node.cpp
    test_pointwise_node.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_fusilli_test(
  NAME fusilli_graph_tests
  SRCS
    test_graph.cpp
    test_context.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_test(
  NAME fusilli_backend_tests
  SRCS
    test_handle.cpp
    test_buffer.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_test(
  NAME fusilli_support_tests
  SRCS
    test_cache.cpp
    test_ssa_validation.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

# Keep test_logging.cpp a separate test target as we manipulate
# logging env vars here which can suppress logs from other tests.
add_fusilli_test(
  NAME fusilli_logging_tests
  SRCS
    test_logging.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_fusilli_lit_test(
  SRC
    lit/test_asm_emitter.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nchw_kcrs.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nchw_krsc.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nhwc_krsc.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nhwc_kcrs.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nchw_kcrs_with_pad.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)

add_fusilli_lit_test(
  SRC
    lit/test_conv_asm_emitter_nhwc_krsc_with_pad.cpp
  DEPS
    libfusilli
  TOOLS
    FileCheck
    iree-opt
    iree-compile
)
