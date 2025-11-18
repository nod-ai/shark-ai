#!/bin/bash -f

export IRPA=/amdshark-dev/llama3.1/405b/fp4/fp4_2025_07_10_fn.irpa
export TOKENIZER=/amdshark-dev/llama3.1/405b/fp4/tokenizer.json
export TOKENIZER_CONFIG=/amdshark-dev/llama3.1/405b/fp4/tokenizer_config.json
SCRIPT_DIR=$(dirname $(realpath "$0"))
export OUTPUT_DIR="${SCRIPT_DIR}/../output_artifacts"
export VMFB=$OUTPUT_DIR/output.vmfb
export CONFIG=$OUTPUT_DIR/config_attn.json
export MODEL="llama-405b-fp4"
export STEPS=64
export CACHE_TYPE="float8_e4m3fn"


OUTPUT_FILE=$OUTPUT_DIR/numeric_validation.log
rm -rf $OUTPUT_FILE

while [[ "$1" != "" ]]; do
    case "$1" in
        --irpa)
                    shift
                    export IRPA=$1
                    ;;
        --vmfb)
                    shift
                    export VMFB=$1
          ;;
        --model)
                    shift
                    export MODEL=$1
          ;;
        --tokenizer)
                    shift
                    export TOKENIZER=$1
          ;;
        --tokenizer_config)
                    shift
                    export TOKENIZER_CONFIG=$1
          ;;
        --config)
                    shift
                    export CONFIG=$1
          ;;
        --steps)
                    shift
                    export STEPS=$1
          ;;
        --kv-cache-dtype)
                    shift
                    export CACHE_TYPE=$1
          ;;
        -h|--help)
                    echo "Usage: $0 [--<different flags>] "
                    echo "--irpa      : path to irpa file"
                    echo "--vmfb            : vmfb file path"
                    echo "--model           : name of the model. "
                    echo "--tokenizer      : Path to tokenizer"
                    echo "--tokenizer_config : Path to tokenizer config"
                    echo "--config      : Path to config"
                    echo "--steps      : Number of decode steps to perform"
                    echo "--kv-cache-dtype      : DType of KV Cache"
                    exit 0
                    ;;
        *)
                    echo "Invalid argument: $1"
                    exit 1
                    ;;
    esac
    shift # Move to the next argument
done


declare -A PROMPT_RESPONSES
# ==================================================================================================
PROMPT_1="<|begin_of_text|>Name the capital of the United States.<|eot_id|>"
RESPONSE_1="The capital of the United States is Washington, D.C."
PROMPT_RESPONSES["$PROMPT_1"]="$RESPONSE_1"

# ==================================================================================================
PROMPT_2="Fire is hot. Yes or No ?"
RESPONSE_2="Yes"
PROMPT_RESPONSES["$PROMPT_2"]="$RESPONSE_2"

# ==================================================================================================

read -r -d '' PROMPT_3 << EOM
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Hey!! Expect the response to be printed as comma separated values.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Give me the first 10 prime numbers<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
EOM

RESPONSE_3="2, 3, 5, 7, 11, 13, 17, 19, 23, 29"

PROMPT_RESPONSES["$PROMPT_3"]="${RESPONSE_3}"


RESULT=0
COUNTER=1

function run_llm_vmfb() {
    local PROMPT=$1
    local RESPONSE=${PROMPT_RESPONSES["$PROMPT"]}
    echo -e "\nExecuting prompt $COUNTER"
    OUTPUT=$(python -m amdsharktank.tools.run_llm_vmfb \
                --prompt "$PROMPT" \
                --irpa $IRPA \
                --vmfb $VMFB \
                --config $CONFIG \
                --tokenizer $TOKENIZER \
                --tokenizer_config $TOKENIZER_CONFIG \
                --steps $STEPS 2>&1)
    printf "%s\n=======================================================\n" | tee -a $OUTPUT_FILE
    printf "%s\nPrompt $COUNTER:\n$PROMPT\n\nResponse: \n$OUTPUT\n\n" | tee -a $OUTPUT_FILE
    RESULT=$(($RESULT || $?))

    case $OUTPUT in
        *"$RESPONSE"* )
            echo "Response matches for prompt $COUNTER"
            ;;
        *)
            echo "Response did not match for prompt $COUNTER"
            RESULT=1
    esac
    ((COUNTER+=1))
    return $RESULT
}


# RUN PROMPT_1
STEPS=20
run_llm_vmfb "$PROMPT_1"
if [[ $RESULT != 0 ]]; then
        echo "Failed to run_llm_vmfb for prompt 1"
fi

# RUN PROMPT_2
STEPS=5
run_llm_vmfb "$PROMPT_2"
if [[ $RESULT != 0 ]]; then
        echo "Failed to run_llm_vmfb for prompt 2"
fi

STEPS=100
run_llm_vmfb "$PROMPT_3"
if [[ $RESULT != 0 ]]; then
        echo "Failed to run_llm_vmfb for prompt 3"
fi


exit $RESULT
