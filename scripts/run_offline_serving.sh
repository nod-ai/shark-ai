#!/bin/bash

export IRPA_PATH=/amdshark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa
export TOKENIZER_JSON=/amdshark-dev/8b/instruct/tokenizer.json
SCRIPT_DIR=$(dirname $(realpath "$0"))
export OUTPUT_DIR="${SCRIPT_DIR}/../output_artifacts"
export VMFB=${OUTPUT_DIR}/output.vmfb
export MODEL_CONFIG=${OUTPUT_DIR}/config_attn.json
export TENSOR_PARALLELISM_SIZE="1"
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
        --tensor-parallelism-size)
            shift
            export TENSOR_PARALLELISM_SIZE=$1
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
            echo "--tensor-parallelism-size : TP size. default 1"
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


RESULTS_DIR=$OUTPUT_DIR/$MODE
mkdir -p $RESULTS_DIR

if [[ $TENSOR_PARALLELISM_SIZE = "8" ]]; then
	set_tp8_parameters
	if [[ $MODE = "all" ]] || [[ $MODE = "chat"  ]]; then
	    echo "Starting offline serving for Chat ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH $IRPA_PATH_RANK0 $IRPA_PATH_RANK1 $IRPA_PATH_RANK2 $IRPA_PATH_RANK3 $IRPA_PATH_RANK4 $IRPA_PATH_RANK5 $IRPA_PATH_RANK6 $IRPA_PATH_RANK7 --benchmark  --benchmark_tasks=16  --device_ids 0 1 2 3 4 5 6 7  --stream --input_token_length 1024 --decode_steps=1024 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
	if [[ $MODE = "all" ]] || [[ $MODE = "reasoning"  ]]; then
	    echo "Starting offline serving for Reasoning ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH  $IRPA_PATH_RANK0 $IRPA_PATH_RANK1 $IRPA_PATH_RANK2 $IRPA_PATH_RANK3 $IRPA_PATH_RANK4 $IRPA_PATH_RANK5 $IRPA_PATH_RANK6 $IRPA_PATH_RANK7 --benchmark  --benchmark_tasks=16  --device_ids 0 1 2 3 4 5 6 7  --stream --input_token_length 1024 --decode_steps=4096 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
	if [[ $MODE = "all" ]] || [[ $MODE = "summary"  ]]; then
	    echo "Starting offline serving for Summary ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH  $IRPA_PATH_RANK0 $IRPA_PATH_RANK1 $IRPA_PATH_RANK2 $IRPA_PATH_RANK3 $IRPA_PATH_RANK4 $IRPA_PATH_RANK5 $IRPA_PATH_RANK6 $IRPA_PATH_RANK7  --benchmark  --benchmark_tasks=16  --device_ids 0 1 2 3 4 5 6 7  --stream --input_token_length 4096 --decode_steps=1024 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
else
	if [[ $MODE = "all" ]] || [[ $MODE = "chat"  ]]; then
	    echo "Starting offline serving for Chat ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=4  --device_ids 0  --stream --input_token_length 1024 --decode_steps=1024 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
	if [[ $MODE = "all" ]] || [[ $MODE = "reasoning"  ]]; then
	    echo "Starting offline serving for Reasoning ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=16  --device_ids 0  --stream --input_token_length 1024 --decode_steps=4096 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
	if [[ $MODE = "all" ]] || [[ $MODE = "summary"  ]]; then
	    echo "Starting offline serving for Summary ..."
	    for conc in $CONCURENCY_LIST ; do
	        python3 -m shortfin_apps.llm.cli --device hip --tokenizer_json=$TOKENIZER_JSON --model_config=$MODEL_CONFIG --vmfb=$VMFB --parameters $IRPA_PATH --benchmark  --benchmark_tasks=16  --device_ids 0  --stream --input_token_length 4096 --decode_steps=1024 --workers_offline=$conc --output-json $RESULTS_DIR/${conc}.json
	    done
	fi
fi
