#!/bin/bash

# This script downloads the necessary files for the Mistral model using aria2c.

FILES=(
  "https://sharkpublic.blob.core.windows.net/sharkpublic/sharkpublic/mistral_nemo/quark/mistral_nemo.irpa /weights/quark/mistral_nemo.irpa"
  # "https://sharkpublic.blob.core.windows.net/sharkpublic/sharkpublic/mistral_nemo/quark/mistral_nemo.json /config/quark/mistral_nemo.json"
  # "https://sharkpublic.blob.core.windows.net/sharkpublic/sharkpublic/mistral_nemo/quark/mistral_nemo.vmfb /config/quark/mistral_nemo.vmfb"
  "https://sharkpublic.blob.core.windows.net/sharkpublic/sharkpublic/mistral_nemo/quark/tokenizer.json /config/quark/tokenizer.json"
  "https://sharkpublic.blob.core.windows.net/sharkpublic/mistral_nemo/tokenizer_config.json /config/quark/tokenizer_config.json"
)

NUM_CONNECTIONS=16
NUM_SPLITS=16

# Loop through the files and download them if they don't exist
for FILE in "${FILES[@]}"; do
  URL=$(echo "$FILE" | awk '{print $1}')
  FILE_PATH=$(echo "$FILE" | awk '{print $2}')

  # Create the directory if it doesn't exist
  mkdir -p $(dirname "$FILE_PATH")

  # Check if the file exists
  if [ ! -f "$FILE_PATH" ]; then
    echo "Downloading $URL to $FILE_PATH"
    aria2c -x "$NUM_CONNECTIONS" -s "$NUM_SPLITS" -d "$(dirname "$FILE_PATH")" -o "$(basename "$FILE_PATH")" "$URL"
    if [ $? -eq 0 ]; then
      echo "Download complete."
    else
      echo "Download failed."
      exit 1
    fi
  else
    echo "$FILE_PATH already exists. Skipping download."
  fi
done

exit 0
