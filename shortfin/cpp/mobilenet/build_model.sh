#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Temporary script to download and build an mnist inference model.
# Eventually, there will be recommended APIs for this but starting
# somewhere.

set -eux
set -o pipefail
TD="$(cd $(dirname $0) && pwd)"

# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.onnx"
ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-v2-10.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12.onnx"
# ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-12.onnx"

mlir_path="$TD/model.mlir"
onnx_path="$TD/model.onnx"
onnx_upgrade_path="$TD/model-17.onnx"
vmfb_path="$TD/model.vmfb"

echo "Downloading to $onnx_path"
curl -L -o $onnx_path "$ONNX_URL"

echo "Converting to version 17"
python $TD/upgrade_onnx.py $onnx_path $onnx_upgrade_path

echo "Import onnx model"
python -m iree.compiler.tools.import_onnx $onnx_upgrade_path -o $mlir_path

echo "Compile onnx model"
if [ -z "$@" ]; then
    compile_flags="--iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host"
else
    compile_flags="$@"
fi
echo "Using compile flags: $compile_flags"
python -m iree.compiler.tools.scripts.iree_compile \
    $mlir_path -o "$vmfb_path" --iree-input-type=onnx $compile_flags
