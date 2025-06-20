#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import onnx
import sys

original_path, converted_path = sys.argv[1:]
original_model = onnx.load_model(original_path)
converted_model = onnx.version_converter.convert_version(original_model, 17)
onnx.save(converted_model, converted_path)
