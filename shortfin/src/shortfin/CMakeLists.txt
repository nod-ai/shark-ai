# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(array)
add_subdirectory(components/tokenizers)
add_subdirectory(components/llm)
add_subdirectory(local)
add_subdirectory(support)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/coroutine)
