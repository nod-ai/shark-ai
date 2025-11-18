#!/bin/bash

## model list

#llama 3.1 Instruct 8B
#llama 3.1 Instruct 70B
#Mistral nemo base 2407
#Mistral nemo instruct 2407
#Flux-Dev
#Flux-Schnell
#SDXL

## input will be
##    1> model name
##    2> prefill BS
##    3> decode BS


export IRPA_PATH=/amdshark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa
export PREFILL_BS="1,2,4,8"
export DECODE_BS="4,8,16,32,64"
export DTYPE="fp16"
SCRIPT_DIR=$(dirname $(realpath "$0"))
export OUTPUT_DIR="${SCRIPT_DIR}/../output_artifacts"
export VMFB=${OUTPUT_DIR}/output.vmfb
export BENCHMARK_DIR=$OUTPUT_DIR/benchmark_module
mkdir -p $BENCHMARK_DIR

while [[ "$1" != "" ]]; do
    case "$1" in
        --parameters)
                    shift
                    export IRPA_PATH=$1
                    ;;
        --vmfb)
                    shift
                    export VMFB=$1
          ;;
        --model)
                    shift
                    export MODEL=$1
          ;;
        --bs-prefill)
                    shift
                    export PREFILL_BS=$1
          ;;
        --bs-decode)
                    shift
                    export DECODE_BS=$1
          ;;
        -h|--help)
                    echo "Usage: $0 [--<different flags>] "
                    echo "--parameters      : path to irpa file"
                    echo "--vmfb            : vmfb file path"
                    echo "--model           : name of the model. "
                    echo "--bs-prefill      : prefill BS"
                    echo "--bs-decode      : prefill BS"
                    exit 0
                    ;;
        *)
                    echo "Invalid argument: $1"
                    exit 1
                    ;;
    esac
    shift # Move to the next argument
done

set_tp8_parameters() {
    irpa_dir_name=$(dirname "$IRPA_PATH")
    echo "irpa_dir_name: $irpa_dir_name"
    irpa_base_name=$(basename "$IRPA_PATH" .irpa)
    echo "irpa_base_name: $irpa_base_name"
    export IRPA_PATH_RANK0=${irpa_dir_name}/${irpa_base_name}.rank0.irpa
    export IRPA_PATH_RANK1=${irpa_dir_name}/${irpa_base_name}.rank1.irpa
    export IRPA_PATH_RANK2=${irpa_dir_name}/${irpa_base_name}.rank2.irpa
    export IRPA_PATH_RANK3=${irpa_dir_name}/${irpa_base_name}.rank3.irpa
    export IRPA_PATH_RANK4=${irpa_dir_name}/${irpa_base_name}.rank4.irpa
    export IRPA_PATH_RANK5=${irpa_dir_name}/${irpa_base_name}.rank5.irpa
    export IRPA_PATH_RANK6=${irpa_dir_name}/${irpa_base_name}.rank6.irpa
    export IRPA_PATH_RANK7=${irpa_dir_name}/${irpa_base_name}.rank7.irpa

    for rank in {0..7}; do
            var_name="IRPA_PATH_RANK$rank"
            echo "irpa_path_rank$rank: ${!var_name}"
    done
}

if [[ $MODEL = "llama-8B-FP8" ]]; then
    echo "$MODEL prefill_bs4 ISL : 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=prefill_bs4 \
        --input=4x128xi64 \
        --input=4xi64 \
        --input=4x4xi64 \
        --input=261x2097152xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP8_prefill_bs4_isl_128.json

    echo "$MODEL decode_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xi64 \
        --input=4xi64 \
        --input=4xi64 \
        --input=4x5xi64 \
        --input=261x2097152xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP8_decode_bs4_isl_128.json

    echo "$MODEL prefill_bs4 ISL : 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=prefill_bs4 \
        --input=4x2048xi64 \
        --input=4xi64 \
        --input=4x64xi64 \
        --input=261x2097152xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP8_prefill_bs4_isl_2048.json

    echo "$MODEL decode_bs4 ISL: 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xi64 \
        --input=4xi64 \
        --input=4xi64 \
        --input=4x65xi64 \
        --input=261x2097152xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP8_decode_bs4_isl_2048.json

elif [[ $MODEL == "llama-70B-FP16" ]]; then

    # FIXME: Use real inputs
    echo "llama-70B-FP16 prefill_bs4 ISL: 128"
    iree-benchmark-module \
          --hip_use_streams=true \
          --module=$VMFB \
          --parameters=model=$IRPA_PATH \
          --device=hip \
          --function=prefill_bs4 \
          --input=4x128xsi64 \
          --input=4xsi64 \
          --input=4x4xsi64 \
          --input=261x5242880xf16 \
          --benchmark_repetitions=3 \
          --benchmark_out_format=json \
          --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16_prefill_bs4_isl_128.json

    echo "llama-70B-FP16 decode_bs4 ISL: 128"
    iree-benchmark-module \
        --hip_use_streams=true \
        --module=$VMFB \
        --parameters=model=$IRPA_PATH \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xsi64 \
        --input=4xsi64 \
        --input=4xsi64 \
        --input=4x5xsi64 \
        --input=261x5242880xf16 \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16_decode_bs4_isl_128.json


    echo "llama-70B-FP16 prefill_bs4 ISL: 2048"
    iree-benchmark-module \
          --hip_use_streams=true \
          --module=$VMFB \
          --parameters=model=$IRPA_PATH \
          --device=hip \
          --function=prefill_bs4 \
          --input=4x2048xsi64 \
          --input=4xsi64 \
          --input=4x64xsi64 \
          --input=513x5242880xf16 \
          --benchmark_repetitions=3 \
          --benchmark_out_format=json \
          --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16_prefill_bs4_isl_2048.json


    echo "llama-70B-FP16 decode_bs4 ISL: 2048"
    iree-benchmark-module \
        --hip_use_streams=true \
        --module=$VMFB \
        --parameters=model=$IRPA_PATH \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xsi64 \
        --input=4xsi64 \
        --input=4xsi64 \
        --input=4x65xsi64 \
        --input=513x5242880xf16 \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16_decode_bs4_isl_2048.json


elif [[ $MODEL == "llama-8B-FP16" ]]; then

    echo "llama-8B-FP16 prefill_bs4 ISL: 128"
    iree-benchmark-module \
          --hip_use_streams=true \
          --module=$VMFB \
          --parameters=model=$IRPA_PATH \
          --device=hip \
          --function=prefill_bs4 \
          --input=@/amdshark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/tokens.npy \
          --input=@/amdshark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_lens.npy \
          --input=@/amdshark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_block_ids.npy \
          --input=@/amdshark-dev/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/cs_f16.npy \
          --benchmark_repetitions=3 \
          --benchmark_out_format=json \
          --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP16_prefill_bs4_isl_128.json

    echo "llama-8B-FP16  decode_bs4 ISL: 128"
    iree-benchmark-module \
      --hip_use_streams=true \
      --module=$VMFB \
      --parameters=model=$IRPA_PATH \
      --device=hip \
      --function=decode_bs4 \
      --input=@/amdshark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/next_tokens.npy \
      --input=@/amdshark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_lens.npy \
      --input=@/amdshark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/start_positions.npy \
      --input=@/amdshark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_block_ids.npy \
      --input=@/amdshark-dev/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/cs_f16.npy \
      --benchmark_repetitions=3 \
      --benchmark_out_format=json \
      --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP16_decode_bs4_isl_128.json

    echo "llama-8B-FP16 prefill_bs4 ISL: 2048"
    iree-benchmark-module \
          --hip_use_streams=true \
          --module=$VMFB \
          --parameters=model=$IRPA_PATH \
          --device=hip \
          --function=prefill_bs4 \
          --input=@/amdshark-dev/8b/prefill_args_bs4_2048_stride_32/tokens.npy \
          --input=@/amdshark-dev/8b/prefill_args_bs4_2048_stride_32/seq_lens.npy \
          --input=@/amdshark-dev/8b/prefill_args_bs4_2048_stride_32/seq_block_ids.npy \
          --input=@/amdshark-dev/8b/prefill_args_bs4_2048_stride_32/cs_f16.npy \
          --benchmark_repetitions=3 \
          --benchmark_out_format=json \
          --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP16_prefill_bs4_isl_2048.json

    echo "llama-8B-FP16 decode_bs4 ISL: 2048"
    iree-benchmark-module \
      --hip_use_streams=true \
      --module=$VMFB \
      --parameters=model=$IRPA_PATH \
      --device=hip \
      --function=decode_bs4 \
      --input=@/amdshark-dev/8b/decode_args_bs4_2048_stride_32/next_tokens.npy \
      --input=@/amdshark-dev/8b/decode_args_bs4_2048_stride_32/seq_lens.npy \
      --input=@/amdshark-dev/8b/decode_args_bs4_2048_stride_32/start_positions.npy \
      --input=@/amdshark-dev/8b/decode_args_bs4_2048_stride_32/seq_block_ids.npy \
      --input=@/amdshark-dev/8b/decode_args_bs4_2048_stride_32/cs_f16.npy \
      --benchmark_repetitions=3 \
      --benchmark_out_format=json \
      --benchmark_out=${BENCHMARK_DIR}/llama-8B-FP16_decode_bs4_isl_2048.json

elif [[ $MODEL == "llama-405B-FP16-tp8" ]]; then
    set_tp8_parameters
    echo "llama-405B-FP16-tp8 prefill_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=prefill_bs4 \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/tokens.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP16-tp8_prefill_bs4_isl_128.json

    echo "llama-405B-FP16-tp8 decode_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=decode_bs4 \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/start_positions.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP16-tp8_decode_bs4_isl_128.json

    echo "llama-405B-FP16-tp8 prefill_bs4 ISL: 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=prefill_bs4 \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/tokens.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/405b/prefill_args_bs4_2048_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP16-tp8_prefill_bs4_isl_2048.json

    echo "llama-405B-FP16-tp8 decode_bs4 ISL: 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=decode_bs4 \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/next_tokens.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/start_positions.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/405b/decode_args_bs4_2048_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP16-tp8_decode_bs4_isl_2048.json

elif [[ $MODEL == "llama-70B-FP16-tp8" ]]; then
    set_tp8_parameters
    echo "llama-70B-FP16-tp8 prefill_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=prefill_bs4 \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/tokens.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/70b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16-tp8_prefill_bs4_isl_128.json

    echo "llama-70B-FP16-tp8 decode_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=decode_bs4 \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/start_positions.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
        --input=@/amdshark-dev/70b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16-tp8_decode_bs4_isl_128.json

    echo "llama-70B-FP16-tp8 prefill_bs4 ISL: 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=prefill_bs4 \
        --input=4x2048xsi64 \
        --input=4xsi64 \
        --input=4x64xsi64 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16-tp8_prefill_bs4_isl_2048.json

    echo "llama-70B-FP16-tp8 decode_bs4 ISL: 2048"
    # TODO: Run real inputs for decode ISL 2048
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --parameters=model="$IRPA_PATH_RANK0" \
        --parameters=model="$IRPA_PATH_RANK1" \
        --parameters=model="$IRPA_PATH_RANK2" \
        --parameters=model="$IRPA_PATH_RANK3" \
        --parameters=model="$IRPA_PATH_RANK4" \
        --parameters=model="$IRPA_PATH_RANK5" \
        --parameters=model="$IRPA_PATH_RANK6" \
        --parameters=model="$IRPA_PATH_RANK7" \
        --device=hip://0 \
        --device=hip://1 \
        --device=hip://2 \
        --device=hip://3 \
        --device=hip://4 \
        --device=hip://5 \
        --device=hip://6 \
        --device=hip://7 \
        --function=decode_bs4 \
        --input=4x1xsi64 \
        --input=4xsi64 \
        --input=4xsi64 \
        --input=4x65xsi64 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --input=513x655360xf16 \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP16-tp8_decode_bs4_isl_2048.json

elif [[ $MODEL == "mistral-nemo-instruct-fp8" ]]; then
    echo "Running prefill BS1 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
      --module=$VMFB    --parameters=model=$IRPA_PATH \
      --function=prefill_bs1   --input=1x2048xsi64   --input=1xsi64 \
      --input=1x32xsi64   --input=2048x2621440xf8E4M3FNUZ \
      --benchmark_repetitions=5 \
      --benchmark_out_format=json \
      --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_prefill_bs1_isl_2048.json

    echo "Running prefill BS2 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=prefill_bs2   --input=2x2048xsi64   --input=2xsi64 \
        --input=2x32xsi64   --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_prefill_bs2_isl_2048.json

    echo "Running prefill BS4 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=prefill_bs4   --input=4x2048xsi64   --input=4xsi64 \
        --input=4x32xsi64   --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_prefill_bs4_isl_2048.json

    echo "Running prefill BS8 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=prefill_bs8   --input=8x2048xsi64   --input=8xsi64 \
        --input=8x32xsi64   --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_prefill_bs8_isl_2048.json

    echo "Running decode BS8 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=decode_bs8   --input=8x1xsi64   --input=8xsi64 \
        --input=8xsi64  --input=8x32xsi64  --input=2048x2621440xf8E4M3FNUZ  \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_decode_bs8_isl_2048.json

    echo "Running decode BS16 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=decode_bs16   --input=16x1xsi64   --input=16xsi64 \
        --input=16xsi64  --input=16x32xsi64  --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_decode_bs16_isl_2048.json

    echo "Running decode BS32 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=decode_bs32   --input=32x1xsi64   --input=32xsi64  \
        --input=32xsi64  --input=32x32xsi64  --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_decode_bs32_isl_2048.json

    echo "Running decode BS64 ISL: 2048"
    iree-benchmark-module   --device=hip   --device_allocator=caching \
        --module=$VMFB    --parameters=model=$IRPA_PATH \
        --function=decode_bs64   --input=64x1xsi64   --input=64xsi64 \
        --input=64xsi64  --input=64x32xsi64  --input=2048x2621440xf8E4M3FNUZ \
        --benchmark_repetitions=5 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/mistral-nemo-instruct-fp8_decode_bs64_isl_2048.json

elif [[ $MODEL == "llama-405B-FP4" ]]; then
    # TODO add more batch size
    echo "llama-405B-FP4 prefill_bs4 ISL: 128"
    input_root_dir="/amdshark-dev/llama3.1/405b/fp4/inputs/real_inputs"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=prefill_bs4 \
        --input=@"${input_root_dir}"/prefill/128/prefill_bs4_tokenlen128_input0_tokens.npy \
        --input=@"${input_root_dir}"/prefill/128/prefill_bs4_tokenlen128_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/prefill/128/prefill_bs4_tokenlen128_input2_seq_block_ids.npy \
        --input=@"${input_root_dir}"/prefill/128/prefill_bs4_tokenlen128_input3_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_prefill_bs4_isl_128.json

    echo "llama-405B-FP4 decode_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=decode_bs4 \
        --input=@"${input_root_dir}"/decode/128/decode_bs4_tokenlen128_input0_tokens.npy \
        --input=@"${input_root_dir}"/decode/128/decode_bs4_tokenlen128_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/decode/128/decode_bs4_tokenlen128_input2_start_positions.npy \
        --input=@"${input_root_dir}"/decode/128/decode_bs4_tokenlen128_input3_seq_block_ids.npy \
        --input=@"${input_root_dir}"/decode/128/decode_bs4_tokenlen128_input4_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_decode_bs4_isl_128.json

    echo "llama-405B-FP4 prefill_bs4 ISL: 512"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=prefill_bs4 \
        --input=@"${input_root_dir}"/prefill/512/prefill_bs4_tokenlen512_input0_tokens.npy \
        --input=@"${input_root_dir}"/prefill/512/prefill_bs4_tokenlen512_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/prefill/512/prefill_bs4_tokenlen512_input2_seq_block_ids.npy \
        --input=@"${input_root_dir}"/prefill/512/prefill_bs4_tokenlen512_input3_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_prefill_bs4_isl_512.json

    echo "llama-405B-FP4 decode_bs4 ISL: 512"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=decode_bs4 \
        --input=@"${input_root_dir}"/decode/512/decode_bs4_tokenlen512_input0_tokens.npy \
        --input=@"${input_root_dir}"/decode/512/decode_bs4_tokenlen512_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/decode/512/decode_bs4_tokenlen512_input2_start_positions.npy \
        --input=@"${input_root_dir}"/decode/512/decode_bs4_tokenlen512_input3_seq_block_ids.npy \
        --input=@"${input_root_dir}"/decode/512/decode_bs4_tokenlen512_input4_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_decode_bs4_isl_512.json

    echo "llama-405B-FP4 prefill_bs4 ISL: 2500"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=prefill_bs4 \
        --input=@"${input_root_dir}"/prefill/2500/prefill_bs4_tokenlen2500_input0_tokens.npy \
        --input=@"${input_root_dir}"/prefill/2500/prefill_bs4_tokenlen2500_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/prefill/2500/prefill_bs4_tokenlen2500_input2_seq_block_ids.npy \
        --input=@"${input_root_dir}"/prefill/2500/prefill_bs4_tokenlen2500_input3_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_prefill_bs4_isl_2500.json

    echo "llama-405B-FP4 decode_bs4 ISL: 2500"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip://0 \
        --function=decode_bs4 \
        --input=@"${input_root_dir}"/decode/2500/decode_bs4_tokenlen2500_input0_tokens.npy \
        --input=@"${input_root_dir}"/decode/2500/decode_bs4_tokenlen2500_input1_seq_lens.npy \
        --input=@"${input_root_dir}"/decode/2500/decode_bs4_tokenlen2500_input2_start_positions.npy \
        --input=@"${input_root_dir}"/decode/2500/decode_bs4_tokenlen2500_input3_seq_block_ids.npy \
        --input=@"${input_root_dir}"/decode/2500/decode_bs4_tokenlen2500_input4_kv_cache_state.npy \
        --benchmark_repetitions=10 \
        -—benchmark_min_warmup_time=10.0 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-405B-FP4_decode_bs4_isl_2500.json

    #TODO:: Add more checks for ISL 5000/10000/20000
elif [[ $MODEL == "llama-70B-FP8" ]]; then
    echo "$MODEL prefill_bs4 ISL : 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=prefill_bs4 \
        --input=4x128xi64 \
        --input=4xi64 \
        --input=4x4xi64 \
        --input=261x5242880xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP8_prefill_bs4_isl_128.json

    echo "$MODEL decode_bs4 ISL: 128"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xi64 \
        --input=4xi64 \
        --input=4xi64 \
        --input=4x5xi64 \
        --input=261x5242880xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP8_decode_bs4_isl_128.json

    echo "$MODEL prefill_bs4 ISL : 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=prefill_bs4 \
        --input=4x2048xi64 \
        --input=4xi64 \
        --input=4x64xi64 \
        --input=261x5242880xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP8_prefill_bs4_isl_2048.json

    echo "$MODEL decode_bs4 ISL: 2048"
    iree-benchmark-module --hip_use_streams=true \
        --module="$VMFB" \
        --parameters=model="$IRPA_PATH" \
        --device=hip \
        --function=decode_bs4 \
        --input=4x1xi64 \
        --input=4xi64 \
        --input=4xi64 \
        --input=4x65xi64 \
        --input=261x5242880xf8E4M3FNUZ \
        --benchmark_repetitions=3 \
        --benchmark_out_format=json \
        --benchmark_out=${BENCHMARK_DIR}/llama-70B-FP8_decode_bs4_isl_2048.json
else
    echo "$MODEL test not implemented"
fi
