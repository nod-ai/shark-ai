#!/usr/bin/env bash

# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Use ${FUSILLI_CACHE_DIR} if set, otherwise default to ${HOME}
CACHE_DIR="${FUSILLI_CACHE_DIR:-$HOME}/.cache/fusilli"

if [ ! -d "${CACHE_DIR}" ]; then
	echo "cache directory ${CACHE_DIR} should exist after running tests"
	exit 1
fi

if [ -n "$(ls -A "${CACHE_DIR}")" ]; then
	echo "cache directory ${CACHE_DIR} should be empty after running tests"
	exit 1
fi
