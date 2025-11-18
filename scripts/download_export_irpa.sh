#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath "$0"))

function download_model() {
    echo "Downloading $MODEL from hugging face"
    if [ -z "${HF_TOKEN}" ]; then
        echo "Hugging face token is empty..please specify using --hf-token"
        exit 1
    fi

    mkdir $MODEL
    hf auth login --token $HF_TOKEN
    if [[ $MODEL = "Llama-3.1-8B-Instruct" || $MODEL = "Llama-3.1-70B-Instruct" ]]; then
        hf download meta-llama/$MODEL --local-dir $MODEL
    elif [[ $MODEL = "Mistral-Nemo-Instruct-2407-FP8" ]]; then
        hf download superbigtree/Mistral-Nemo-Instruct-2407-FP8 --local-dir $MODEL
    fi
}

function convert_to_gguf() {
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    python3 convert_hf_to_gguf.py ../$MODEL/ --outtype f16 --outfile ../$MODEL/${MODEL}.gguf
}

function convert_from_gguf_to_irpa() {
    python -m amdsharktank.tools.dump_gguf --gguf-file ../$MODEL/${MODEL}.gguf --output-irpa ../$MODEL/${MODEL}.irpa

}

while [[ "$1" != "" ]]; do
    case "$1" in
    --model)
        shift
        export MODEL=$1
        ;;
    --hf-token)
        shift
        export HF_TOKEN=$1
        ;;
    -h | --help)
        echo "Usage: $0 [--<different flags>] "
        echo "--model            : Model to run (Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Mistral-Nemo-Instruct-2407-FP8 )"
        echo "--hf-token        : Hugging face token with access to gated flux models"
        exit 0
        ;;
    *)
        echo "Invalid argument: $1"
        exit 1
        ;;
    esac
    shift # Move to the next argument
done

download_model $MODEL $HF_TOKEN

if [[ $? = 0 ]]; then
    if [[ $MODEL = "Mistral-Nemo-Instruct-2407-FP8" ]]; then
        python scripts/merge_safetensors.py $MODEL
        if [[ $? = 0 ]]; then
            python -m amdsharktank.models.llama.tools.import_quark_dataset --params merged.safetensors --output-irpa-file=$MODEL/$MODEL.irpa \
                 --config-json $MODEL/config.json --model-base="70b" --weight-dtype=float16
            if [[ $? = 0 ]]; then
                date=$(date -u +'%Y-%m-%d')
                sudo cp /amdshark-dev/mistral_instruct/instruct.irpa /amdshark-dev/mistral_instruct/instruct.irpa_${date}
                sudo cp $MODEL/${MODEL}.irpa /amdshark-dev/mistral_instruct/instruct.irpa
            else
                echo "IRPA export for $MODEL failed"
            fi
        else
            echo "Merging of safetensors failed"
        fi
    else
        convert_to_gguf $MODEL
        if [[ $? = 0 ]]; then
            convert_from_gguf_to_irpa $MODEL
            if [[ $? = 0 ]]; then
                if [[ $MODEL = "Llama-3.1-8B-Instruct" ]]; then
                    date=$(date -u +'%Y-%m-%d')
                    sudo cp /amdshark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa /amdshark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa_${date}
                    sudo cp ../$MODEL/${MODEL}.irpa /amdshark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa
                    cd ..
                fi
                if [[ $MODEL = "Llama-3.1-70B-Instruct" ]]; then
                    date=$(date -u +'%Y-%m-%d')
                    sudo cp /amdshark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa /amdshark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa_${date}
                    sudo cp ../$MODEL/${MODEL}.irpa /amdshark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa
                    cd ..
                fi
            else
                echo "Conversion from gguf to IRPA failed for $MODEL"
                exit 1
            fi
        else
            echo "Conversion to gguf failed for $MODEL"
            exit 1
        fi
    fi
else
    echo "Downloading of $MODEL failed"
    exit 1
fi
