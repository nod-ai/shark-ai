#!/bin/bash
# fetch from huggingface
set -e

huggingface_path=$1
model_name=$2
export_dtype="bf16"
destination="${HOME}/.cache/shark/genfiles/flux"

mkdir -p tmp
cd tmp

# Export
## Export weights to irpa
echo "Starting export of parameters"
python -m sharktank.pipelines.flux.export_parameters --dtype $export_dtype --input-path $huggingface_path --output-path ./ --model $model_name

# Move files to cache
mkdir -p $destination
# Copy VMFB files
# Fallback to t5xxl irpa fetching during server startup
rm ./exported_parameters_${export_dtype}/flux_dev_t5xxl_${export_dtype}.irpa # TODO: Modify T5 to enable initialization from the flux repo
# Copy IRPA files from exported_parameters directory
mv exported_parameters_${export_dtype}/* "$destination/"

echo "Flux export complete. Files copied to $destination"

cd ..
rm -rf tmp
