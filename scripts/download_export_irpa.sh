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

    # Determine repository based on model name
    if [ -n "${HF_REPO}" ]; then
        REPO=$HF_REPO
    elif [[ $MODEL = "Llama-3.1-8B-Instruct" || $MODEL = "Llama-3.1-70B-Instruct" ]]; then
        REPO="meta-llama/$MODEL"
    elif [[ $MODEL = "Llama-3.1-8B-Instruct-FP8-KV" ]]; then
        REPO="amd/Llama-3.1-8B-Instruct-FP8-KV"
    elif [[ $MODEL = "Mistral-Nemo-Instruct-2407-FP8" ]]; then
        REPO="superbigtree/Mistral-Nemo-Instruct-2407-FP8"
    else
        # Default: assume model name contains the full repo path (e.g., "owner/model-name")
        REPO=$MODEL
    fi

    echo "Downloading from repository: $REPO"
    hf download $REPO --local-dir $MODEL
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
    --hf-repo)
        shift
        export HF_REPO=$1
        ;;
    -h | --help)
        echo "Usage: $0 [--<different flags>] "
        echo "--model            : Model to run (Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Llama-3.1-8B-Instruct-FP8-KV, Mistral-Nemo-Instruct-2407-FP8 )"
        echo "--hf-token         : Hugging face token with access to gated flux models"
        echo "--hf-repo          : (Optional) Hugging Face repository in format 'owner/repo-name'. If not specified, will be inferred from model name"
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
    if [[ $MODEL = "Llama-3.1-8B-Instruct-FP8-KV" ]]; then
        if [ -f "$MODEL/merged.safetensors" ]; then
            rm "$MODEL/merged.safetensors"
        fi
        python scripts/merge_safetensors.py $MODEL
        sudo mv merged.safetensors $MODEL/merged.safetensors
        if [[ $? = 0 ]]; then
            python -m sharktank.models.llama.tools.import_quark_dataset \
                --params $MODEL/merged.safetensors \
                --output-irpa-file=instruct_8b_fp8_e4m3fn.irpa \
                --config-json $MODEL/config.json \
                --model-base="7b" \
                --quantizer-dtype float8_e4m3fnuz \
                --weight-dtype-override float16
            if [[ $? = 0 ]]; then
                echo "IRPA export for $MODEL completed successfully: instruct_8b_fp8_e4m3fn.irpa"
            else
                echo "IRPA export for $MODEL failed"
                exit 1
            fi
        else
            echo "Merging of safetensors failed"
            exit 1
        fi
    elif [[ $MODEL = "Mistral-Nemo-Instruct-2407-FP8" ]]; then
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
