#!/bin/bash

# Default batch sizes from build config
DEFAULT_PREFILL_BS="1,2,4"
DEFAULT_DECODE_BS="8,16,24,32,64,128"

# Parse command line arguments
PREFILL_BS=${1:-$DEFAULT_PREFILL_BS}
DECODE_BS=${2:-$DEFAULT_DECODE_BS}

echo "Using prefill batch sizes: $PREFILL_BS"
echo "Using decode batch sizes: $DECODE_BS"

echo "Step 1: Building VMFB"
# Compile the config
if [ ! -f "/config/quark/mistral_nemo.vmfb" ]; then
  echo "Compiling the config"
  python -m sharktank.examples.export_paged_llm_v1 \
    --irpa-file /weights/quark/mistral_nemo.irpa \
    --output-mlir /config/quark/mistral_nemo.mlir \
    --output-config /config/quark/mistral_nemo.json \
    --bs-prefill=$PREFILL_BS \
    --bs-decode=$DECODE_BS \
    --activation-dtype=float16 \
    --attention-dtype=float16 \
    --use-hf \
    --attention-kernel=torch \
    --kv-cache-dtype=float8_e4m3fnuz \
    --device-block-count 4096 && \
    iree-compile /config/quark/mistral_nemo.mlir \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx942 \
    --iree-opt-level=O3  \
    --iree-hal-indirect-command-buffers=true  \
    --iree-stream-resource-memory-model=discrete  \
    --iree-hal-memoization=true \
    --iree-codegen-enable-default-tuning-specs=true \
    -o /config/quark/mistral_nemo.vmfb
else
  echo "Config already compiled. Skipping compilation."
fi

echo "Step 2: Running Benchmarks"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "SOURCE_DIR: $SOURCE_DIR"

# Split comma-separated values into arrays
IFS=',' read -ra PREFILL_BS_ARR <<< "$PREFILL_BS"
IFS=',' read -ra DECODE_BS_ARR <<< "$DECODE_BS"

# Run prefill benchmarks
echo "Running prefill benchmarks for batch sizes: $PREFILL_BS"
for bs in "${PREFILL_BS_ARR[@]}"; do
    echo "Running prefill benchmark with batch size: $bs"
    iree-benchmark-module --function=prefill_bs${bs} \
        --module=$CONFIG_DIR/mistral_nemo.vmfb \
        --device=hip:0 \
        --input=${bs}x1024xi64=0 \
        --input=${bs}xi64=1022 \
        --input=${bs}x32xi64=0 \
        --input=16x2621440xf8E4M3FNUZ=0 \
        --output=@outputs/prefill_bs${bs}_out0.npy \
        --parameters=model=$SOURCE_DIR/mistral_nemo.irpa
done

# Run decode benchmarks
echo "Running decode benchmarks for batch sizes: $DECODE_BS"
for bs in "${DECODE_BS_ARR[@]}"; do
    echo "Running decode benchmark with batch size: $bs"
    iree-benchmark-module --function=decode_bs${bs} \
        --module=$CONFIG_DIR/mistral_nemo.vmfb \
        --device=hip:0 \
        --input=${bs}x1xi64=0 \
        --input=${bs}xi64=2022 \
        --input=${bs}xi64=2022 \
        --input=${bs}x64xi64=0 \
        --input=32x2621440xf8E4M3FNUZ=0 \
        --output=@outputs/decode_${bs}_out0.npy \
        --parameters=model=$SOURCE_DIR/mistral_nemo.irpa
done
