#!/bin/bash

set -eux

export ROCR_VISIBLE_DEVICES=5

mkdir -p /home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits
TRACE_PREFIX=/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits \
python -m sharktank.tools.eval_llm_model \
    "--irpa=/shark-dev/llama3.1/8b/instruct/weights/instruct_8b_fp8_e4m3fnuz.irpa" \
    "--tokenizer=/shark-dev/llama3.1/8b/instruct" \
    "--dataset=/home/bpetkant/ws/sharktank/repo/sharktank/tests/evaluate/datasets/llama_8b_fp8_e4m3_fnuz_torch.json" \
    "--expected-err=1" \
    "--min-context=10"

mkdir -p /home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-import-preset-logits
TRACE_PREFIX=/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-import-preset-logits \
python -m sharktank.tools.eval_llm_model \
    "--irpa=/home/bpetkant/ws/sharktank/experiments/llama3/export/llama3.1/8b/instruct/f8_e4m3fnuz/model.irpa" \
    "--tokenizer=/home/bpetkant/ws/sharktank/experiments/llama3/export/llama3.1/8b/instruct/f8_e4m3fnuz" \
    "--dataset=/home/bpetkant/ws/sharktank/repo/sharktank/tests/evaluate/datasets/llama_8b_fp8_e4m3_fnuz_torch.json" \
    "--expected-err=1" \
    "--min-context=10"

mkdir -p /home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits
TRACE_PREFIX=/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits \
python -m sharktank.tools.eval_llm_model \
    "--irpa=/shark-dev/llama3.1/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa" \
    "--tokenizer=/shark-dev/llama3.1/8b/instruct" \
    "--dataset=/home/bpetkant/ws/sharktank/repo/sharktank/tests/evaluate/datasets/llama_8b_fp16_torch.json" \
    "--expected-err=1e-2" \
    "--min-context=10"

mkdir -p /home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits
TRACE_PREFIX=/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits \
python -m sharktank.tools.eval_llm_model \
    "--irpa=/home/bpetkant/ws/sharktank/experiments/llama3/export/llama3.1/8b/instruct/f16/model.irpa" \
    "--tokenizer=/home/bpetkant/ws/sharktank/experiments/llama3/export/llama3.1/8b/instruct/f16" \
    "--dataset=/home/bpetkant/ws/sharktank/repo/sharktank/tests/evaluate/datasets/llama_8b_fp16_torch.json" \
    "--expected-err=1" \
    "--min-context=10"
