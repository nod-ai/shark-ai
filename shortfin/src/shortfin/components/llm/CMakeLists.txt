# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

shortfin_cc_component(
  NAME
    shortfin_llm_components
  HDRS
    beam_processor.h
  SRCS
    beam_processor.cc
  COMPONENTS
    shortfin_support
)

shortfin_gtest_test(
  NAME beam_processor_test
  SRCS
    beam_processor_test.cc
)
