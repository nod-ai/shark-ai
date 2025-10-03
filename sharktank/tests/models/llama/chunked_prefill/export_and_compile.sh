#!/bin/bash

set -eux

#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

python3 -m sharktank.examples.export_paged_llm_v1 \
    --irpa-file=/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa \
    --output-mlir=model.mlir \
    --output-config=model.json \
    --bs-prefill=4 \
    --bs-decode=4 \
    --has-prefill-position \
    --device-block-count 4096

iree-compile model.mlir \
    --iree-hip-target=gfx942 -o model.vmfb \
    --iree-hal-target-device=hip \
    --iree-opt-level=O3 \
    --iree-opt-strip-assertions=0 \
    --iree-vm-c-module-strip-debug-ops=0 \
    --iree-vm-bytecode-module-strip-debug-ops=0 \
    --iree-hal-indirect-command-buffers=true \
    --iree-stream-resource-memory-model=discrete \
    --iree-hip-enable-tensor-ukernels \
    --iree-stream-affinity-solver-max-iterations=1024 \
    --iree-hal-memoization=true --iree-codegen-enable-default-tuning-specs=true \
    --iree-llvmgpu-test-combine-layout-transformation=false
