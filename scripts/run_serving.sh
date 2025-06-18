#!/bin/bash

export IRPA_PATH=/sharedfile/attn/fp8_attn.irpa
export TOKENIZER_JSON=/shark-dev/8b/instruct/tokenizer.json
export VMFB=$(pwd)/../output_artifacts/output.vmfb
export MODEL_CONFIG=$(pwd)/../output_artifacts/config_attn.json
export PORT=8959


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
        --port)
            shift
            export port=$1
            ;;
        -h|--help)
            echo "Usage: $0 [--<different flags>] "
            echo "--irpa            : path to irpa file"
            echo "--tokenizer_json  : model tokenizer json file path "
            echo "--vmfb            : vmfb file path"
            echo "--port            : port number on which client-server will run. default: 8959"
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

echo "Running server ..."

python -m shortfin_apps.llm.server \
           --tokenizer_json=$TOKENIZER_JSON \
           --model_config=$MODEL_CONFIG \
           --vmfb=$VMFB \
           --parameters=$IRPA_PATH \
           --device=hip \
           --device_ids 0  --port $port & 

shortfin_process=$!

echo $shortfin_process

sleep 10
echo "Running Client ..."

curl http://localhost:$port/generate \
           -H "Content-Type: application/json" \
           -d '{
              "text": "<|begin_of_text|>Name the capital of the United States.<|eot_id|>",
                "sampling_params": {"max_completion_tokens": 50}
            }' > inference_output

sleep 10
kill -9 $shortfin_process

