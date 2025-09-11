#!/bin/bash
set -e
trap 'echo "Error occurred on command: $BASH_COMMAND"' ERR

export IRPA_PATH=/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa
export PREFILL_BS="1,2,4,8"
export DECODE_BS="4,8,16,32,64"
export DTYPE="fp16"
export TENSOR_PARALLELISM_SIZE="1"
SCRIPT_DIR=$(dirname $(realpath "$0"))
export OUTPUT_DIR="${SCRIPT_DIR}/../output_artifacts"

while [[ "$1" != "" ]]; do
    case "$1" in
        --irpa)
            shift
            export IRPA_PATH=$1
            ;;
        --bs-prefill)
            shift
            export PREFILL_BS=$1
            ;;
        --bs-decode)
            shift
            export DECODE_BS=$1
            ;;
        --dtype)
            shift
            export DTYPE=$1
            if [[ "$DTYPE" = "fp8" ]]; then
                export ATTENTION_DTYPE="float16"
                export ACTIVATION_DTYPE="float16"
                export KV_CACHE_DTYPE="float8_e4m3fnuz"
            fi
            ;;
        --tensor-parallelism-size)
            shift
            export TENSOR_PARALLELISM_SIZE=$1
            ;;
        --output_dir)
            shift
            export OUTPUT_DIR=$1
            ;;
        -h|--help)
            echo "Usage: $0 [--<different flags>] "
            echo "--irpa        : path to irpa file"
            echo "--bs-prefill  : prefill BS to be exported. Default: 1,2,4,8"
            echo "--bs-decode   : decode BS to be exported. Default: 4,8,16,32,64"
            echo "--dtype       : Data type to be used. Default: fp16"
            echo "--output_dir  : Absolute path of directory for dumping the artifacts. Default: '\$PWD/output_artifacts' "
            exit 0
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

start=$(date +%s)
echo "### Exporting IR .... "
mkdir -p $OUTPUT_DIR

if [[ $DTYPE = "fp8" ]]; then
    python3 -m sharktank.examples.export_paged_llm_v1 --irpa-file=$IRPA_PATH \
        --output-mlir=$OUTPUT_DIR/output.mlir \
        --output-config=$OUTPUT_DIR/config_attn.json \
        --bs-prefill=$PREFILL_BS --bs-decode=$DECODE_BS --attention-kernel sharktank \
        --use-hf --device-block-count 8043
elif [[ $DTYPE = "mistral_fp8" ]]; then
    python3 -m sharktank.examples.export_paged_llm_v1 --irpa-file=$IRPA_PATH \
        --output-mlir=$OUTPUT_DIR/output.mlir \
        --output-config=$OUTPUT_DIR/config_attn.json \
        --bs-prefill=$PREFILL_BS --bs-decode=$DECODE_BS \
        --use-hf --attention-kernel=torch --device-block-count 4096

elif [[ $TENSOR_PARALLELISM_SIZE = "8" ]]; then
    python3 -m sharktank.examples.export_paged_llm_v1  --irpa-file=$IRPA_PATH \
        --output-mlir=$OUTPUT_DIR/output.mlir \
        --output-config=$OUTPUT_DIR/config_attn.json \
        --bs-prefill=$PREFILL_BS --bs-decode=$DECODE_BS  --device-block-count 32768 \
        --tensor-parallelism-size=$TENSOR_PARALLELISM_SIZE
else
    python3 -m sharktank.examples.export_paged_llm_v1  --irpa-file=$IRPA_PATH \
        --output-mlir=$OUTPUT_DIR/output.mlir \
        --output-config=$OUTPUT_DIR/config_attn.json \
        --bs-prefill=$PREFILL_BS --bs-decode=$DECODE_BS  --device-block-count 4096
fi
end=$(date +%s)
echo "Time taken for exporting: $((end - start)) seconds"

start=$(date +%s)
echo "### compiling IR .... "

if [[ $TENSOR_PARALLELISM_SIZE = "8" ]]; then
    iree-compile $OUTPUT_DIR/output.mlir \
        --iree-hip-target=gfx942 -o $OUTPUT_DIR/output.vmfb \
        --iree-hal-target-device="hip[0]" \
        --iree-hal-target-device="hip[1]" \
        --iree-hal-target-device="hip[2]" \
        --iree-hal-target-device="hip[3]" \
        --iree-hal-target-device="hip[4]" \
        --iree-hal-target-device="hip[5]" \
        --iree-hal-target-device="hip[6]" \
        --iree-hal-target-device="hip[7]" \
        --iree-opt-level=O3 \
        --iree-hal-indirect-command-buffers=true \
        --iree-stream-resource-memory-model=discrete \
        --iree-hal-memoization=true --iree-codegen-enable-default-tuning-specs=true \
        --iree-hip-enable-tensor-ukernels \
        --iree-stream-affinity-solver-max-iterations=1024
else
    iree-compile $OUTPUT_DIR/output.mlir \
        --iree-hip-target=gfx942 -o $OUTPUT_DIR/output.vmfb \
        --iree-hal-target-device=hip --iree-opt-level=O3 \
        --iree-hal-indirect-command-buffers=true \
        --iree-stream-resource-memory-model=discrete \
        --iree-hip-enable-tensor-ukernels \
        --iree-stream-affinity-solver-max-iterations=1024 \
        --iree-hal-memoization=true --iree-codegen-enable-default-tuning-specs=true
fi

end=$(date +%s)
echo "Time taken for compiling: $((end - start)) seconds"
