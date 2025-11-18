#!/bin/bash

export IRPA_PATH=/sharedfile/attn/fp8_attn.irpa
export TOKENIZER_JSON=/amdshark-dev/8b/instruct/tokenizer.json
export VMFB=$(pwd)/../output_artifacts/output.vmfb
export MODEL_CONFIG=$(pwd)/../output_artifacts/config_attn.json
export port=8959
export TENSOR_PARALLELISM_SIZE=1
SCRIPT_DIR=$(dirname $(realpath "$0"))
source ${SCRIPT_DIR}/server_utils.sh


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
        --tensor-parallelism-size)
            shift
            export TENSOR_PARALLELISM_SIZE=$1
            ;;
        -h|--help)
            echo "Usage: $0 [--<different flags>] "
            echo "--irpa            : path to irpa file"
            echo "--tokenizer_json  : model tokenizer json file path "
            echo "--tensor-parallelism-size : TP size. default 1"
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

echo "Running server ..."
if [[ $TENSOR_PARALLELISM_SIZE = "8" ]]; then
	set_tp8_parameters
	python -m shortfin_apps.llm.server \
	           --tokenizer_json=$TOKENIZER_JSON \
	           --model_config=$MODEL_CONFIG \
	           --vmfb=$VMFB \
	           --parameters $IRPA_PATH $IRPA_PATH_RANK0 $IRPA_PATH_RANK1 $IRPA_PATH_RANK2 $IRPA_PATH_RANK3 $IRPA_PATH_RANK4 $IRPA_PATH_RANK5 $IRPA_PATH_RANK6 $IRPA_PATH_RANK7 \
	           --device=hip \
	           --device_ids 0 1 2 3 4 5 6 7  --port $port &
	shortfin_process=$!
else
	python -m shortfin_apps.llm.server \
	           --tokenizer_json=$TOKENIZER_JSON \
	           --model_config=$MODEL_CONFIG \
	           --vmfb=$VMFB \
	           --parameters=$IRPA_PATH \
	           --device=hip \
	           --device_ids 0  --port $port &
	shortfin_process=$!
fi

wait_for_server $port

if [[ ! -e /proc/$shortfin_process ]]; then
    echo "Failed to start the server"
    exit 1
fi

echo "Server with PID $shortfin_process is ready to accept requests on port $port....."

echo "Running Client ..."

start_time=$(date +%s)

curl http://localhost:$port/generate \
           -H "Content-Type: application/json" \
           -d '{
              "text": "<|begin_of_text|>Name the capital of the United States.<|eot_id|>",
                "sampling_params": {"max_completion_tokens": 50}
            }' > ${SCRIPT_DIR}/../output_artifacts/online_serving.log

end_time=$(date +%s)
time_taken=$((end_time - start_time))
echo -e "\nTime Taken for Getting Response: $time_taken seconds" >> ${SCRIPT_DIR}/../output_artifacts/online_serving.log

sleep 10
kill -9 $shortfin_process

# Check if the file exists
file="${SCRIPT_DIR}/../output_artifacts/online_serving.log"
if [ -e "$file" ]; then
    echo "The file '$file' exists."
else
    echo "The file '$file' does NOT exist."
    exit 1
fi

# Check for Online Serving Response
expected_1="\"responses\": [{\"text\": \"assistant\\nThe capital of the United States is Washington, D.C.\"}]"
expected_2="\"responses\": [{\"text\": \"assistant\\n\\nThe capital of the United States is Washington, D.C. (short for District of Columbia).\"}]"
expected_3="\"responses\": [{\"text\": \"assistant\\n\\nThe capital of the United States is Washington, D.C.\"}]"
expected_4="\"responses\": [{\"text\": \"Washington D.C.\"}]"

if grep -F "$expected_1" "$file"; then
    echo "[SUCCESS] Online Response Matches Expected Output."
elif grep -F "$expected_2" "$file"; then
    echo "[SUCCESS] Online Response Matches Expected Output."
elif grep -F "$expected_3" "$file"; then
    echo "[SUCCESS] Online Response Matches Expected Output."
elif grep -F "$expected_4" "$file"; then
    echo "[SUCCESS] Online Response Matches Expected Output."
elif grep -Eiq '"text": ".*washington(,?\s*d\.?c\.?)?"' "$file"; then
    echo "[CHECK REQUIRED] Partially Correct Response Detected."
    cat "$file"
    exit 1
else
    echo "[FAILURE] Gibberish or Invalid Response Detected."
    cat "$file"
    exit 1
fi
