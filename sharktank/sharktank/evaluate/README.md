# LLM Evaluation Pipeline

## Setup
Setup SHARK Platform's Evaluation Pipeline

```
pip install -r sharktank/requirements-tests.txt
```

### Perplexity

Perplexity score measures the ability of a language model to predict the next token in a sequence. A lower score indicates that a model has higher certainty in it's predictions. Perplexity acts as an intrinsic evaluation metric that measures the model quality, independent of any downstream task.

In SHARK-Platform, we use perplexity to track code regressions and quality loss across quantized models (with FP16 as baseline). We use 100 prompts from the Wikitext-2 test set and calculate the mean perplexities shown below. These numbers are neither comparable between models with different tokenizers nor with other projects due to varying implementations.

Test perplexity for Llama3.1 8B & 405B (FP16 & FP8) models:

```bash
pytest sharktank/tests/evaluate/perplexity_test.py  --longrun
```

Calculate the perplexity for a new model:

```bash
python -m  sharktank.evaluate.perplexity \
  --gguf-file=llama3_70b_f16.gguf \
  --tokenizer-config-json=tokenizer_config.json
```

### LLaMA 3.1 Scoreboard

| CPU            | GPU        |
|:---------------|:-----------|
| AMD EPYC 9554  | MI300X     |


|Models   |Model size (GB) |Torch      |IREE       |
|:--------|:---------------|:----------|:----------|
|8B f16   |16.07           |14.930181  |14.991893  |
