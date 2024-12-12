# LLama 8b GPU instructions on MI300X

## Setup

We will use an example with `llama_8b_f16` in order to describe the
process of exporting a model for use in the shortfin llm server with an
MI300 GPU.

### Pre-Requisites

- Python >= 3.11 is recommended for this flow
    - You can check out [pyenv](https://github.com/pyenv/pyenv)
    as a good tool to be able to manage multiple versions of python
    on the same system.

### Create virtual environment

To start, create a new virtual environment:

```bash
python -m venv --prompt shark-ai .venv
source .venv/bin/activate
```

### Install `shortfin`

You can install either the `latest stable` version of shortfin by installing
`shark-ai` or the `nightly` version directly:

#### Stable

```bash
pip install shark-ai[apps]
```

#### Nightly

```bash
pip install shortfin[apps] --pre -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
```

<!-- TODO: Remove when sharktank added to `shark-ai[apps]` -->
### Install `sharktank`

Install the `nightly` version of sharktank:

```bash
pip install sharktank --pre -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
```

<!-- TODO: Remove once `sentencepiece` added to nightly `sharktank` -->
### Install `sentencepiece`

```bash
pip install sentencepiece
```

### Define a directory for export files

Create a new directory for us to export files like
`model.mlir`, `model.vmfb`, etc.

```bash
mkdir $PWD/export
export EXPORT_DIR=$PWD/export
```

### Download llama3_8b_fp16.gguf

We will use the `hf_datasets` module in `sharktank` to download a
LLama3.1 8b f16 model.

```bash
python -m sharktank.utils.hf_datasets llama3_8B_fp16 --local-dir $EXPORT_DIR
```

### Define environment variables

Define the following environment variables to make running
this example a bit easier:

#### Model/Tokenizer vars

This example uses the `llama8b_f16.gguf` and `tokenizer.json` files
that were downloaded in the previous step.

```bash
export MODEL_PARAMS_PATH=$EXPORT_DIR/meta-llama-3.1-8b-instruct.f16.gguf
export TOKENIZER_PATH=$EXPORT_DIR/tokenizer.json
```

#### General env vars

The following env vars can be copy + pasted directly:

```bash
# Path to export model.mlir file
export MLIR_PATH=$EXPORT_DIR/model.mlir
# Path to export config.json file
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json
# Path to export model.vmfb file
export VMFB_PATH=$EXPORT_DIR/model.vmfb
# Batch size for kvcache
export BS=1,4
# NOTE: This is temporary, until multi-device is fixed
export ROCR_VISIBLE_DEVICES=1
```

## Export to MLIR

We will now use the `sharktank.examples.export_paged_llm_v1` script
to export our model to `.mlir` format.

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs=$BS
```

## Compiling to `.vmfb`

Now that we have generated a `model.mlir` file,
we can compile it to `.vmfb` format, which is required for running
the `shortfin` LLM server.

We will use the
[iree-compile](https://iree.dev/developers/general/developer-overview/#iree-compile)
tool for compiling our model.

### Compile for MI300

**NOTE: This command is specific to MI300 GPUs.
For other `--iree-hip-target` GPU options,
look [here](https://iree.dev/guides/deployment-configurations/gpu-rocm/#compile-a-program)**

```bash
iree-compile $MLIR_PATH \
 --iree-hal-target-backends=rocm \
 --iree-hip-target=gfx942 \
 -o $VMFB_PATH
```

## Running the `shortfin` LLM server

We should now have all of the files that we need to run the shortfin LLM server.

Verify that you have the following in your specified directory ($EXPORT_DIR):

```bash
ls $EXPORT_DIR
```

- config.json
- meta-llama-3.1-8b-instruct.f16.gguf
- model.mlir
- model.vmfb
- tokenizer_config.json
- tokenizer.json

### Launch server

#### Run the shortfin server

Now that we are finished with setup, we can start the Shortfin LLM Server.

Run the following command to launch the Shortfin LLM Server in the background:

> **Note**
> By default, our server will start at `http://localhost:8000`.
> You can specify the `--host` and/or `--port` arguments, to run at a different address.
>
> If you receive an error similar to the following:
>
> `[errno 98] address already in use`
>
> Then, you can confirm the port is in use with `ss -ntl | grep 8000`
> and either kill the process running at that port,
> or start the shortfin server at a different port.

```bash
python -m shortfin_apps.llm.server \
   --tokenizer_json=$TOKENIZER_PATH \
   --model_config=$OUTPUT_CONFIG_PATH \
   --vmfb=$VMFB_PATH \
   --parameters=$MODEL_PARAMS_PATH \
   --device=hip > shortfin_llm_server.log 2>&1 &
shortfin_process=$!
```

You can verify your command has launched successfully
when you see the following logs outputted to terminal:

```bash
cat shortfin_llm_server.log
```

#### Expected output

```text
[2024-10-24 15:40:27.440] [info] [on.py:62] Application startup complete.
[2024-10-24 15:40:27.444] [info] [server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Verify server

We can now verify our LLM server by sending a simple request:

### Open python shell

```bash
python
```

### Send request

```python
import requests

import os

port = 8000 # Change if running on a different port

generate_url = f"http://localhost:{port}/generate"

def generation_request():
    payload = {"text": "Name the capital of the United States.", "sampling_params": {"max_completion_tokens": 50}}
    try:
        resp = requests.post(generate_url, json=payload)
        resp.raise_for_status()  # Raises an HTTPError for bad responses
        print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

generation_request()
```

After you receive the request, you can exit the python shell:

```bash
quit()
```

## Cleanup

When done, you can kill the shortfin_llm_server by killing the process:

```bash
kill -9 $shortfin_process
```
