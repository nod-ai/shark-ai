#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e  # Exit on any error

# Script to export, compile, validate, and upload IRPA files for LLama models
# This script consolidates common workflow steps for different model configurations

# Help function
show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Export, compile, validate, and upload IRPA files for LLama models.

Required Options:
  --model-tag TAG               Model identifier tag (e.g., llama3_8b_fp16)
  --hf-model MODEL              HuggingFace model name to download
  --hf-token TOKEN              HuggingFace authentication token
  --irpa-path PATH              Path to the IRPA file
  --irpa-filename FILENAME      IRPA filename for Azure storage
  --kv-cache-dtype DTYPE        KV cache data type (e.g., float16, float8_e4m3fnuz)
  --tokenizer-path PATH         Path to tokenizer.json file
  --tokenizer-config-path PATH  Path to tokenizer_config.json file
  --azure-blob-path PATH        Azure blob storage path (e.g., ossci-models/llama_3_1)
  --azure-sas-token TOKEN       Azure SAS token for authentication
  --date-suffix SUFFIX          Date suffix for versioned uploads

Optional Options:
  --dtype DTYPE                 Model data type (e.g., fp8, fp16)
  --bs-prefill NUM              Prefill batch size (default: 4)
  --bs-decode NUM               Decode batch size (default: 4)
  --steps NUM                   Number of validation steps (default: 64)
  --help                        Show this help message

EOF
}

# Default values
BS_PREFILL=4
BS_DECODE=4
STEPS=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      show_help
      exit 0
      ;;
    --model-tag)
      MODEL_TAG="$2"
      shift 2
      ;;
    --hf-model)
      HF_MODEL="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --irpa-path)
      IRPA_PATH="$2"
      shift 2
      ;;
    --irpa-filename)
      IRPA_FILENAME="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --tokenizer-config-path)
      TOKENIZER_CONFIG_PATH="$2"
      shift 2
      ;;
    --azure-blob-path)
      AZURE_BLOB_PATH="$2"
      shift 2
      ;;
    --azure-sas-token)
      AZURE_SAS_TOKEN="$2"
      shift 2
      ;;
    --date-suffix)
      DATE_SUFFIX="$2"
      shift 2
      ;;
    --bs-prefill)
      BS_PREFILL="$2"
      shift 2
      ;;
    --bs-decode)
      BS_DECODE="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$MODEL_TAG" ] || [ -z "$HF_MODEL" ] || [ -z "$HF_TOKEN" ] || [ -z "$IRPA_PATH" ] || \
   [ -z "$IRPA_FILENAME" ] || [ -z "$KV_CACHE_DTYPE" ] || [ -z "$TOKENIZER_PATH" ] || \
   [ -z "$TOKENIZER_CONFIG_PATH" ] || [ -z "$AZURE_BLOB_PATH" ] || [ -z "$AZURE_SAS_TOKEN" ] || \
   [ -z "$DATE_SUFFIX" ]; then
  echo "Error: Missing required parameters"
  echo "Required: --model-tag, --hf-model, --hf-token, --irpa-path, --irpa-filename,"
  echo "          --kv-cache-dtype, --tokenizer-path, --tokenizer-config-path,"
  echo "          --azure-blob-path, --azure-sas-token, --date-suffix"
  exit 1
fi

echo "MODEL_TAG: $MODEL_TAG"

# Export model
echo "=== Exporting $MODEL_TAG model ==="
bash scripts/download_export_irpa.sh \
  --model "$HF_MODEL" \
  --hf-token "$HF_TOKEN" || { echo "Export failed"; exit 1; }

# Run export and compile
echo "=== Running export and compile ==="
COMPILE_CMD="bash scripts/export_and_compile.sh --irpa $IRPA_PATH --bs-prefill $BS_PREFILL --bs-decode $BS_DECODE"
if [ -n "$DTYPE" ]; then
  COMPILE_CMD="$COMPILE_CMD --dtype $DTYPE"
fi
$COMPILE_CMD 2>&1 | tee "$(pwd)/output_artifacts/${MODEL_TAG}_export_and_compilation.log" || { echo "Compilation failed"; exit 1; }

# Validate VMFB Responses
echo "=== Validating VMFB Responses ==="
bash scripts/validate_numerics.sh \
  --irpa "$IRPA_PATH" \
  --vmfb "$(pwd)/output_artifacts/output.vmfb" \
  --config "$(pwd)/output_artifacts/config_attn.json" \
  --tokenizer "$TOKENIZER_PATH" \
  --tokenizer_config "$TOKENIZER_CONFIG_PATH" \
  --steps "$STEPS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" | tee "$(pwd)/output_artifacts/${MODEL_TAG}_run_llm_vmfb.log" || { echo "Validation failed"; }

# Check for IRPA changes
echo "=== Checking for IRPA changes ==="
echo "Downloading latest IRPA file from Azure"
PREVIOUS_IRPA="${IRPA_FILENAME%.irpa}_previous.irpa"
az storage blob download \
  --account-name sharkpublic \
  --sas-token "$AZURE_SAS_TOKEN" \
  --container-name ossci \
  --name "$AZURE_BLOB_PATH/$IRPA_FILENAME" \
  --file "$PREVIOUS_IRPA" \
  --no-progress || echo "No previous IRPA file found, will upload new file"

UPLOAD_REQUIRED=false
if [ -f "$PREVIOUS_IRPA" ]; then
  echo "Comparing IRPA files"
  if ! diff -q "$IRPA_PATH" "$PREVIOUS_IRPA" > /dev/null 2>&1; then
    echo "IRPA files differ, upload required"
    UPLOAD_REQUIRED=true
  else
    echo "IRPA files are identical, skipping upload"
  fi
else
  echo "No previous IRPA file found, upload required"
  UPLOAD_REQUIRED=true
fi

# Upload IRPA file if required
if [ "$UPLOAD_REQUIRED" = true ]; then
  echo "=== Uploading new IRPA for $MODEL_TAG ==="
  # Upload with date suffix
  az storage blob upload \
    --account-name sharkpublic \
    --sas-token "$AZURE_SAS_TOKEN" \
    --container-name ossci \
    --name "$AZURE_BLOB_PATH/${IRPA_FILENAME%.irpa}-${DATE_SUFFIX}.irpa" \
    --file "$IRPA_PATH"

  # Upload current version (overwrite)
  az storage blob upload \
    --account-name sharkpublic \
    --sas-token "$AZURE_SAS_TOKEN" \
    --container-name ossci \
    --name "$AZURE_BLOB_PATH/$IRPA_FILENAME" \
    --file "$IRPA_PATH" \
    --overwrite
fi

echo "=== Completed $MODEL_TAG workflow ==="
