#!/bin/bash

export IRPA_PATH=/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa
export TOKENIZER_JSON=/shark-dev/8b/instruct/tokenizer.json
export VMFB=$(pwd)/../output_artifacts/output.vmfb
export MODEL_CONFIG=$(pwd)/../output_artifacts/config_attn.json
export MODE=all
export CONCURENCY_LIST="4 8 16 32 64 128 256"


while [[ "$1" != "" ]]; do
    case "$1" in
        --irpa)
            shift
            export IRPA_PATH=$1
            ;;
        --tokenizer_json)
            shift
            export TOKENIZER_JSON=$1
            ;;
        --vmfb)
            shift
            export VMFB=$1
            ;;
        --model_config)
            shift
            export MODEL_CONFIG=$1
            ;;
        --mode)
            shift
            export MODE=$1
            ;;
        -h|--help)
            echo "Usage: $0 [--<different flags>] "
            echo "--irpa            : path to irpa file"
            echo "--tokenizer_json  : model tokenizer json file path "
            echo "--vmfb            : vmfb file path"
            echo "--mode            : chat|reasoning|summary|all"
            echo "--model_config    : model config json file path"
            exit 0
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done


if [[ $MODE = "all" ]] || [[ $MODE = "chat"  ]]; then
    echo "Starting offline serving for Chat ..."
    for conc in $CONCURENCY_LIST ; do
        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=16  --device_ids 0  --stream --input_token_length 1024 --decode_steps=1024 --workers_offline=$conc
    done
fi
if [[ $MODE = "all" ]] || [[ $MODE = "reasoning"  ]]; then
    echo "Starting offline serving for Reasoning ..."
    for conc in $CONCURENCY_LIST ; do
        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=16  --device_ids 0  --stream --input_token_length 1024 --decode_steps=4096 --workers_offline=$conc
    done
fi
if [[ $MODE = "all" ]] || [[ $MODE = "summary"  ]]; then
    echo "Starting offline serving for Summary ..."
    for conc in $CONCURENCY_LIST ; do
        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=16  --device_ids 0  --stream --input_token_length 4096 --decode_steps=1024 --workers_offline=$conc
    done
fi
