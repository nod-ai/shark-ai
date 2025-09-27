# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/bin/bash

# example: ./upload_goldens.sh --config Wan2_1_14B_T2V_512x512x20 --dir .

set -e

# --- Default values ---
CONFIG=""
SOURCE_DIR="."

# --- Parse Command Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift # past argument
            ;;
        --dir)
            SOURCE_DIR="$2"
            shift # past argument
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # past value
done

# --- Validate required arguments ---
if [ -z "$CONFIG" ]; then
    echo "Error: --config option is required."
    echo "Usage: $0 --config <model_config> [--dir <source_directory>]"
    exit 1
fi

ACCOUNT_NAME="sharkpublic"
DESTINATION_CONTAINER="sharkpublic"
BASE_PATH="wan/${CONFIG}/goldens"

CURRENT_DATE=$(date +%m%d%Y)

FULL_DESTINATION_PATH="${DESTINATION_CONTAINER}/${BASE_PATH}/${CURRENT_DATE}"

FILE_PATTERN="*put.safetensors"

echo "Starting Azure Storage upload..."
echo "---------------------------------"
echo "Account:     ${ACCOUNT_NAME}"
echo "Config:      ${CONFIG}"
echo "Destination: ${FULL_DESTINATION_PATH}"
echo "Pattern:     '${FILE_PATTERN}'"
echo "Source:      '${SOURCE_DIR}'"
echo "---------------------------------"

az storage blob upload-batch \
  --account-name "${ACCOUNT_NAME}" \
  --destination "${FULL_DESTINATION_PATH}" \
  --source "${SOURCE_DIR}" \
  --pattern "${FILE_PATTERN}" \
  --overwrite

echo ""
echo "Upload complete!"
