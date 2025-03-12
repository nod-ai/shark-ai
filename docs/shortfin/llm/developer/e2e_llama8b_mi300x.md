# LLama 8b GPU Instructions on MI300X

**NOTE: This was ran on the `mi300x-3` system**

## Setup

We will use an example with `llama_8b_f16_decomposed` in order to describe the
process of exporting a model for use in the shortfin llm server with an MI300 GPU.

### Pre-Requisites

- Python >= 3.11 is recommended for this flow
    - You can check out [pyenv](https://github.com/pyenv/pyenv) as a good tool
    to be able to manage multiple versions of python on the same system.

### Setting Up Environment

Follow the `Development Getting Started` docs
[here](https://github.com/nod-ai/shark-ai/blob/main/README.md#development-getting-started)
to setup your environment for development.

We will use an example with `llama_8b_f16_decomposed` in order to describe the
process of exporting a model for use in the shortfin llm server with an MI300 GPU.

### Define a directory for export files

Create a new directory for us to export files like `model.mlir`, `model.vmfb`, etc.

```bash
mkdir $PWD/export
export EXPORT_DIR=$PWD/exportd
```

### Define environment variables

Define the following environment variables to make running this example a bit easier:

#### Model/Tokenizer vars

This example uses the `llama8b_f16.irpa` and `tokenizer.json` files that are
pre-existing on the MI300X-3 system.
You may need to change the paths for your own system.

```bash
export MODEL_PARAMS_PATH=/data/llama3.1/8b/llama8b_f16.irpa # Path to existing .irpa file, may need to change w/ system
export TOKENIZER_PATH=/data/llama3.1/8b/tokenizer.json # Path to existing tokenizer.json, may need to change w/ system
```

#### General env vars

The following env vars can be copy + pasted directly:

```bash
export MLIR_PATH=$EXPORT_DIR/model.mlir # Path to export model.mlir file
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json # Path to export config.json file
export EDITED_CONFIG_PATH=$EXPORT_DIR/edited_config.json # Path to export config.json file
export VMFB_PATH=$EXPORT_DIR/model.vmfb # Path to export model.vmfb file
export BS=1,4 # Batch size for kvcache
export ROCR_VISIBLE_DEVICES=1 # NOTE: This is temporary, until multi-device is fixed
```

### Export to MLIR

We will now use the `sharktank.examples.export_paged_llm_v1` script to export
our model to `.mlir` format.

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --irpa-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs-prefill=$BS
  --bs-decode=$BS
```

## Compiling to `.vmfb`

Now that we have generated a `model.mlir` file, we can compile it to `.vmfb`
format, which is required for running the `shortfin` LLM server.

We will use the [iree-compile](https://iree.dev/developers/general/developer-overview/#iree-compile)
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

## Write an edited config

We need to write a config for our model with a slightly edited structure
to run with shortfin. This will work for the example in our docs.
You may need to modify some of the parameters for a specific model.

### Write edited config

```bash
cat > $EDITED_CONFIG_PATH << EOF
{
    "module_name": "module",
    "module_abi_version": 1,
    "max_seq_len": 131072,
    "attn_head_count": 8,
    "attn_head_dim": 128,
    "prefill_batch_sizes": [
        $BS
    ],
    "decode_batch_sizes": [
        $BS
    ],
    "transformer_block_count": 32,
    "paged_kv_cache": {
        "block_seq_stride": 16,
        "device_block_count": 256
    }
}
EOF
```

## Running the `shortfin` LLM server

We should now have all of the files that we need to run the shortfin LLM server.

Verify that you have the following in your specified directory ($EXPORT_DIR):

```bash
ls $EXPORT_DIR
```

- edited_config.json
- model.vmfb

### Launch server:

#### Set the target device

<!-- TODO: Add instructions on targeting different devices,
when `--device=hip://$DEVICE` is supported -->

#### Run the shortfin server

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
   --model_config=$EDITED_CONFIG_PATH \
   --vmfb=$VMFB_PATH \
   --parameters=$MODEL_PARAMS_PATH \
   --device=hip > shortfin_llm_server.log 2>&1 &
shortfin_process=$!
```

You can verify your command has launched successfully when you see the following
 logs outputted to terminal:

```bash
cat shortfin_llm_server.log
```

#### Expected output

```text
[2024-10-24 15:40:27.440] [info] [on.py:62] Application startup complete.
[2024-10-24 15:40:27.444] [info] [server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Verify server

### Client script

We can test the LLM server, by running our client script:

```bash
python shortfin/python/shortfin_apps/llm/client.py --port 8000
```

### Simple request

Or by sending a simple request:

### Open python shell

```bash
python
```

### Send request

```python
import requests

import os

port = 8000 # Change if running at a different port

generate_url = f"http://localhost:{port}/generate"

def generation_request():
    payload = {"text": "What is the capital of the United States?", "sampling_params": {"max_completion_tokens": 50}}
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
