# Llama end to end serving instructions

## Introduction

This guide demonstrates how to serve the
[Llama family](https://www.llama.com/) of Large Language Models (LLMs) using
shark-ai.

* By the end of this guide you will have a server running locally and you will
  be able to send HTTP requests containing chat prompts and receive chat
  responses back.

* We will demonstrate the development flow using a version of the
  [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  model, quantized to fp16. Other models in the
  [Llama 3.1 family](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
  are supported as well.

Overview:

1. Setup, installing dependencies and configuring the environment
2. Download model files then compile the model for our accelerator(s) of choice
3. Start a server using the compiled model files
4. Send chat requests to the server and receive chat responses back

## 1. Setup

### Pre-requisites

- An installed
  [AMD Instinct™ MI300X Series Accelerator](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
    - Other accelerators should work too, but shark-ai is currently most
      optimized on MI300X
- Compatible versions of Linux and ROCm (see the [ROCm compatability matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html))
- Python >= 3.11

### Create virtual environment

To start, create a new
[virtual environment](https://docs.python.org/3/library/venv.html):

```bash
python -m venv --prompt shark-ai .venv
source .venv/bin/activate
```

### Install Python packages

Install `shark-ai`, which includes the `sharktank` model development toolkit
and the `shortfin` serving framework:

```bash
pip install shark-ai[apps]
```

> [!TIP]
> To switch from the stable release channel to the nightly release channel,
> see [`nightly_releases.md`](../../../nightly_releases.md).

The `sharktank` project contains implementations of popular LLMs optimized for
ahead of time compilation and serving via `shortfin`. These implementations are
built using PyTorch, so install a `torch` version that fulfills your needs by
following either https://pytorch.org/get-started/locally/ or our recommendation:

```bash
# Fast installation of torch with just CPU support.
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Prepare a working directory

Create a new directory for model files and compilation artifacts:

```bash
export EXPORT_DIR=$PWD/export
mkdir -p $EXPORT_DIR
```

## 2. Download and compile the model

### Download `llama3_8b_fp16.gguf`

We will use the `hf_datasets` module in `sharktank` to download a
LLama3.1 8b f16 model.

```bash
python -m sharktank.utils.hf_datasets llama3_8B_fp16 --local-dir $EXPORT_DIR
```

> [!NOTE]
> If you have the model weights as a collection of `.safetensors` files (downloaded from HuggingFace Model Hub, for example), you can use the `convert_hf_to_gguf.py` script from the [llama.cpp repository](https://github.com/ggerganov/llama.cpp) to convert them to a single `.gguf` file.
> ```bash
> export WEIGHTS_DIR=/path/to/safetensors/weights_directory/
> git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
> cd llama.cpp
> python3 convert_hf_to_gguf.py $WEIGHTS_DIR --outtype f16 --outfile $EXPORT_DIR/<output_gguf_name>.gguf
> ```
> Now this GGUF file can be used in the instructions ahead.

### Define environment variables

We'll first define some environment variables that are shared between the
following steps.

#### Model/tokenizer variables

This example uses the `llama8b_f16.gguf` and `tokenizer.json` files
that were downloaded in the previous step.

```bash
export MODEL_PARAMS_PATH=$EXPORT_DIR/meta-llama-3.1-8b-instruct.f16.gguf
export TOKENIZER_PATH=$EXPORT_DIR/tokenizer.json
```

#### General environment variables

These variables configure the model export and compilation process:

```bash
export MLIR_PATH=$EXPORT_DIR/model.mlir
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json
export VMFB_PATH=$EXPORT_DIR/model.vmfb
export EXPORT_BATCH_SIZES=1,4
# NOTE: This is temporary, until multi-device is fixed
export ROCR_VISIBLE_DEVICES=1
```

### Export to MLIR using sharktank

We will now use the
[`sharktank.examples.export_paged_llm_v1`](https://github.com/nod-ai/shark-ai/blob/main/sharktank/sharktank/examples/export_paged_llm_v1.py)
script to export an optimized implementation of the LLM from PyTorch to the
`.mlir` format that our compiler can work with:

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs=$EXPORT_BATCH_SIZES
```

### Compile using IREE to a `.vmfb` file

Now that we have generated a `model.mlir` file, we can compile it to the `.vmfb`
format, which is required for running the `shortfin` LLM server. We will use the
[iree-compile](https://iree.dev/developers/general/developer-overview/#iree-compile)
tool for compiling our model.

```bash
iree-compile $MLIR_PATH \
 --iree-hal-target-backends=rocm \
 --iree-hip-target=gfx942 \
 -o $VMFB_PATH
```

> [!NOTE]
> The `--iree-hip-target=gfx942` option will generate code for MI300 series
> GPUs. To compile for other targets, see
> [the options here](https://iree.dev/guides/deployment-configurations/gpu-rocm/#compile-a-program).

### Check exported files

We should now have all of the files that we need to run the shortfin LLM server:

```bash
ls -1A $EXPORT_DIR
```

Expected output:

```
config.json
meta-llama-3.1-8b-instruct.f16.gguf
model.mlir
model.vmfb
tokenizer_config.json
tokenizer.json
```

## 3. Run the `shortfin` LLM server

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

Expected output:

```
[2024-10-24 15:40:27.440] [info] [on.py:62] Application startup complete.
[2024-10-24 15:40:27.444] [info] [server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 4. Test the server

We can now test our LLM server. First let's confirm that it is running:

```bash
curl -i http://localhost:8000/health

# HTTP/1.1 200 OK
# date: Thu, 19 Dec 2024 19:40:43 GMT
# server: uvicorn
# content-length: 0
```

Next, let's send a generation request:

```bash
curl http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Name the capital of the United States.",
        "sampling_params": {"max_completion_tokens": 50}
    }'
```

The response should come back as `Washington, D.C.!`.

### Send requests from Python

You can also send HTTP requests from Python like so:

```python
import os
import requests

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

## Cleanup

When done, you can stop the `shortfin_llm_server` by killing the process:

```bash
kill -9 $shortfin_process
```

If you want to find the process again:

```bash
ps -f | grep shortfin
```

## Server Options

To run the server with different options, you can use the
following command to see the available flags:

```bash
python -m shortfin_apps.llm.server --help
```

### Server Options

A full list of options can be found below:

| Argument                                        | Description                                                                                                                                                                         |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host HOST`                                   | Specify the host to bind the server.                                                                                                                                                |
| `--port PORT`                                   | Specify the port to bind the server.                                                                                                                                                |
| `--root-path ROOT_PATH`                         | Root path to use for installing behind a path-based proxy.                                                                                                                          |
| `--timeout-keep-alive TIMEOUT_KEEP_ALIVE`       | Keep-alive timeout duration.                                                                                                                                                        |
| `--tokenizer_json TOKENIZER_JSON`               | Path to a `tokenizer.json` file.                                                                                                                                                    |
| `--tokenizer_config_json TOKENIZER_CONFIG_JSON` | Path to a `tokenizer_config.json` file.                                                                                                                                             |
| `--model_config MODEL_CONFIG`                   | Path to the model config file.                                                                                                                                                      |
| `--vmfb VMFB`                                   | Model [VMFB](https://iree.dev/developers/general/developer-tips/#inspecting-vmfb-files) to load.                                                                                    |
| `--parameters [FILE ...]`                       | Parameter archives to load (supports: `gguf`, `irpa`, `safetensors`).                                                                                                               |
| `--device {local-task,hip,amdgpu}`              | Device to serve on (e.g., `local-task`, `hip`). Same options as [iree-run-module --list_drivers](https://iree.dev/guides/deployment-configurations/gpu-rocm/#get-the-iree-runtime). |
| `--device_ids [DEVICE_IDS ...]`                 | Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a device ID like `amdgpu:0:0@0`.                                                   |
| `--isolation {none,per_fiber,per_call}`         | Concurrency control: How to isolate programs.                                                                                                                                       |
| `--amdgpu_async_allocations`                    | Enable asynchronous allocations for AMD GPU device contexts.                                                                                                                        |
