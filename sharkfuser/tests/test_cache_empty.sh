#! /bin/bash

# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if [ ! -d "$HOME/.cache/fusilli" ]; then
	echo "cache directory $HOME/.cache/fusilli should exist after running tests"
	exit 1
fi

if [ -n "$(ls -A "$HOME/.cache/fusilli")" ]; then
	echo "cache directory $HOME/.cache/fusilli should be empty after running tests"
	exit 1
fi
