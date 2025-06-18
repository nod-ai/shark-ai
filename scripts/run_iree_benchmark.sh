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
##	1> model name
##	2> prefill BS
##	3> decode BS


export IRPA_PATH=/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa
export PREFILL_BS="1,2,4,8"
export DECODE_BS="8,16,32,64"
export DTYPE="fp16"
export OUTPUT_DIR="$(pwd)/output_artifacts"
export VMFB=$(pwd)/output_artifacts/output.vmfb


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

if [[ $MODEL = "llama-8B-FP8" ]]; then
	echo "$MODEL prefill_bs4"
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
	
	echo "$MODEL decode_bs4"
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
		--benchmark_repetitions=3
else
	echo "$MODEL Not Implemented"
fi
