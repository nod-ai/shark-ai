
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

shortfin_cc_component(
  NAME
    shortfin_llm_components
  HDRS
    data.h
    selectors.h
  SRCS
    selectors.cc

  COMPONENTS
    shortfin_support
)

set_property(GLOBAL APPEND
  PROPERTY SHORTFIN_LIB_OPTIONAL_COMPONENTS
  shortfin_llm_components)
