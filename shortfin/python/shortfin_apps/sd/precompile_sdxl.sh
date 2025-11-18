#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [--model_weights <path>] [--model_json <file>] [--flag_file <file>] [--td_spec <file>] [--force_export <True|False>] [--gpu_batch_size <int>] [--vae_batch_size <int>] [--quant_path <path>]"
    exit 1
}

# Default values
model_weights=""
output_dir="~/.cache/amdshark"
model_json="./examples/sdxl_config_fp8_sched_unet.json"
flag_file="./examples/sdxl_flags_gfx942.txt"
td_spec=""
force_export=false
gpu_batch_size=""
vae_batch_size=""
quant_path=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_weights)
            model_weights="$2"; shift 2;;
        --model_json)
            model_json="$2"; shift 2;;
        --flag_file)
            flag_file="$2"; shift 2;;
        --td_spec)
            td_spec="$2"; shift 2;;
        --force_export)
            force_export="$2"; shift 2;;
        --gpu_batch_size)
            gpu_batch_size="$2"; shift 2;;
        --vae_batch_size)
            vae_batch_size="$2"; shift 2;;
        --quant_path)
            quant_path="$2"; shift 2;;
        *)
            usage;;
    esac
done

# Set script paths
script_dir="$(dirname "$(realpath "$0")")"
script_path="$script_dir/$model_json"
flagfile="$script_dir/$flag_file"
if [[ -n "$td_spec" ]]; then
    td_spec="$script_dir/$td_spec"
fi
shortfin_dir=$script_dir
export IREE_BUILD_MP_CONTEXT="fork"

# Modify JSON batch sizes
sed -i -E "s/\"clip\": \[[0-9]+\]/\"clip\": [$gpu_batch_size]/g; \
           s/\"scheduled_unet\": \[[0-9]+\]/\"scheduled_unet\": [$gpu_batch_size]/g; \
           s/\"vae\": \[[0-9]+\]/\"vae\": [${vae_batch_size:-$gpu_batch_size}]/g" "$script_path"

# Parse model_flags from flag file
current_model="all"
declare -A model_flags
model_flags["all"]=""

while IFS= read -r line; do
    if [[ "$line" == --* ]]; then
        model_flags["$current_model"]+="$line "
    else
        current_model="$line"
        model_flags["$current_model"]=""
    fi
done < "$flagfile"

# Append td_spec if provided
if [[ -n "$td_spec" ]]; then
    echo "Applying TD spec"
        for key in "unet" "punet" "scheduled_unet" "vae"; do
        if [[ -n "${model_flags[$key]}" ]]; then
            model_flags[$key]+=" --iree-codegen-transform-dialect-library=$td_spec"
        fi
    done
fi

# Execute iree.build commands
for modelname in "clip" "scheduled_unet" "vae"; do
    batch_size="$gpu_batch_size"
    if [[ "$modelname" == "vae" && -n "$vae_batch_size" ]]; then
        batch_size="$vae_batch_size"
    fi

    ireec_extra_args="${model_flags[all]} ${model_flags[$modelname]}"

    builder_args=(
        python -m iree.build "$shortfin_dir/components/builders.py"
        "--model-json=$script_path"
        "--target=gfx942"
        "--splat=false"
        "--build-preference=export"
        "--output-dir=$output_dir"
        "--model=$modelname"
        "--force-update=$force_export"
        "--iree-hal-target-device=amdgpu"
        "--iree-hip-target=gfx942"
        "--iree-compile-extra-args=$ireec_extra_args"
        "--quant-path=$quant_path"
    )

    echo "Executing: ${builder_args[*]}"
    output=$("${builder_args[@]}")

done
