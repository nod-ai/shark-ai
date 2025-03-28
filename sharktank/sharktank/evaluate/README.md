# LLM Evaluation Pipeline

## Setup
Setup SHARK Platform's Evaluation Pipeline

```
pip install -r sharktank/requirements-tests.txt
```

### Perplexity

Perplexity score measures the ability of a language model to predict the next token in a sequence. A lower score indicates that a model has higher certainty in it's predictions. Perplexity acts as an intrinsic evaluation metric that measures the model quality, independent of any downstream task.

In SHARK-Platform, we use perplexity to track code regressions and quality loss across quantized models (with FP16 as baseline). We use 100 prompts randomly selected from the Wikitext-2 test set and calculate the mean perplexities shown below. These numbers are neither comparable between models with different tokenizers nor with other projects due to varying implementations.

* Test perplexity for Llama3.1 8B (FP16) model:

```bash
pytest sharktank/tests/evaluate/perplexity_test.py  --longrun
```

* Calculate perplexity for a new model:

```bash
pytest -n 8 -v -s sharktank/tests/evaluate/perplexity_iree_test.py -k test_llama3_8B_f16 \
  --llama3-8b-f16-model-path=llama3.1_8b_instruct_fp16.irpa  \
  --llama3-8b-tokenizer-path=tokenizer_config.json \
  --bs=4 \
  --iree-device=hip://0 \
  --iree-hip-target=gfx942 \
  --iree-hal-target-device=hip
```

For a new model:

Replace `--irpa-file` with `--gguf-file` flag if necessary (eg: `--gguf-file=llama3_70b_instruct_fp16.gguf`)

##### Torch mode
```bash
python -m  sharktank.evaluate.perplexity_torch \
  --irpa-file=llama3_70b_instruct_fp16.irpa \
  --tokenizer-config-json=tokenizer_config.json \
  --num-prompts=4 \
  --device='cuda:0'
```

##### IREE mode

To run on MI300:
```bash
python -m sharktank.evaluate.perplexity_iree \
  --irpa-file=llama3_70b_instruct_fp16.irpa \
  --tokenizer-config-json=tokenizer_config.json \
  --num-prompts=4 \
  --iree-device='hip://0' \
  --iree-hal-target-device=hip \
  --iree-hip-target=gfx942
```

To run on CPU, replace the above --iree-* flags with:
```bash
  --iree-device='local-task' --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu
```

For additional options:
```bash
python -m sharktank.evaluate.perplexity_torch  -h
python -m sharktank.evaluate.perplexity_iree  -h
```

### Perplexity Scoreboard

| CPU            | GPU        |
|:-------------: |:----------:|
| AMD EPYC 9554  | MI300X     |

#### LLaMA 3.1

|Models                 |Model size (GB) |Torch score   |IREE score    |
|:----------------------|:---------------|:-------------|:-------------|
|8B FP16 TP1 decomposed |16.07           |14.930181     |14.991893     |
