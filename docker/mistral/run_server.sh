#!/bin/bash

echo "Running server"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "SOURCE_DIR: $SOURCE_DIR"

# ROCR_VISIBLE_DEVICES=0 python -m shortfin_apps.llm.server \
#     --tokenizer_json $CONFIG_DIR/tokenizer.json \
#     --model_config $CONFIG_DIR/mistral_nemo.json \
#     --vmfb $CONFIG_DIR/mistral_nemo.vmfb \
#     --device=hip \
#     --device_ids 0 \
#     --parameters $SOURCE_DIR/mistral_nemo.irpa \
#     --num_beams 8 \
#     --port 8000

ROCR_VISIBLE_DEVICES=0 python -m shortfin_apps.llm.cli \
    --device hip \
    --tokenizer_json=$CONFIG_DIR/tokenizer.json  \
    --model_config=$CONFIG_DIR/mistral_nemo.json  \
    --vmfb=$CONFIG_DIR/mistral_nemo.vmfb    \
    --parameters $SOURCE_DIR/mistral_nemo.irpa  \
    --decode_steps=64  \
    --num_beams=8 \
    --device_ids=0 \
    --benchmark  \
    --benchmark_tasks=32 \
    --workers_offline=8  \
    --input_token_length=2500 \
    --log_tokens \
    --top_k=8
