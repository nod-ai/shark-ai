# LLM Evaluation Pipeline

## Setup
Setup SHARK Platform's Evaluation Pipeline

```
pip install -r sharktank/requirements-tests.txt
```

### Perplexity

Perplexity score measures the ability of a language model to predict the next token in a sequence. A lower score indicates that a model has higher certainty in it's predictions. Perplexity acts as an intrinsic evaluation metric that measures the model quality, independent of any downstream task.

In SHARK-Platform, we use perplexity to track code regressions and quality loss across quantized models (with FP16 as baseline). We use 100 prompts randomly selected from the Wikitext-2 test set and calculate the mean perplexities shown below. These numbers are neither comparable between models with different tokenizers nor with other projects due to varying implementations.

* Test perplexity for Llama3.1 8B (FP16) model on a MI300 server:

#### Torch/eager mode

```bash
pytest -n 8 -v -s sharktank/tests/evaluate/perplexity_torch_test.py -k test_llama3_8B_f16 \
  --llama3-8b-f16-model-path=llama3.1_8b_instruct_fp16.irpa \
  --llama3-8b-tokenizer-path=tokenizer_config.json \
  --bs=4
```

#### IREE mode

```bash
pytest -n 8 -v -s sharktank/tests/evaluate/perplexity_iree_test.py -k test_llama3_8B_f16 \
  --llama3-8b-f16-model-path=llama3.1_8b_instruct_fp16.irpa  \
  --llama3-8b-tokenizer-path=tokenizer_config.json \
  --bs=4 \
  --iree-device=hip://1 \
  --iree-hip-target=gfx942 \
  --iree-hal-target-device=hip \

```

* Calculate perplexity for a new model:

Replace `irpa` with `gguf` files if necessary (eg: `--gguf-file=llama3_70b_instruct_fp16.gguf`)

#### Torch/eager mode Perplexity:

```bash
python -m  sharktank.evaluate.perplexity_torch \
  --irpa-file=llama3_70b_instruct_fp16.irpa \
  --tokenizer-config-json=tokenizer_config.json \
  --num-prompts=10
```

For additional options:

```bash
python -m sharktank.evaluate.perplexity_torch  -h
```

#### IREE Perplexity:

To run on MI300:

```bash
python -m sharktank.evaluate.perplexity_iree \
  --irpa-file=llama3_70b_instruct_fp16.irpa \
  --tokenizer-config-json=tokenizer_config.json \
  --num-prompts=10 \
  --iree-device='hip://0' \
  --iree-hal-target-device=hip \
  --iree-hip-target=gfx942
```

To run on CPU, replace the above --iree-* flags with:

```bash
  --iree-device='local-task' \
  --iree-hal-target-device=llvm-cpu
```

For additional options:

```bash
python -m sharktank.evaluate.perplexity_iree  -h
```

### Perplexity Scoreboard

| CPU            | GPU        | Num of prompts   |
|:-------------: |:----------:|:----------------:|
| AMD EPYC 9554  | MI300X     |      100         |

#### LLaMA 3.1

|Models                          |Torch score   |IREE score    | Model size (GB) |
|:-------------------------------|:-------------|:-------------|:----------------|
|8B FP16 Instruct TP1            |20.303255     |19.786807     |16.07            |
