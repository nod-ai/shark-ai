# amdshark User Guide

These instructions cover the usage of the latest stable release of amdshark. For a more bleeding edge release please install the [nightly releases](nightly_releases.md).

> [!TIP]
> Please note as we are prepping the next stable release, please use [nightly releases](nightly_releases.md) for usage.

## Prerequisites

Our current user guide requires that you have:
- Access to a computer with an installed AMD Instinct™ MI300x Series Accelerator
- Installed a compatible version of Linux and ROCm on the computer (see the [ROCm compatability matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html))

## Set up Environment

This section will help you install Python and set up a Python environment with venv.

Officially we support Python versions: 3.11, 3.12, 3.13

The rest of this guide assumes you are using Python 3.11.

### Install Python

To install Python 3.11 on Ubuntu:

```bash
sudo apt install python3.11 python3.11-dev python3.11-venv

which python3.11
# /usr/bin/python3.11
```

### Create a Python Environment

Setup your Python environment with the following commands:

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.11 -m venv 3.11.venv
source 3.11.venv/bin/activate
```

## Install amdshark and its dependencies

First install a torch version that fulfills your needs:

```bash
# Fast installation of torch with just CPU support.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For other options, see https://pytorch.org/get-started/locally/.

Next install amdshark-ai:

```bash
pip install amdshark-ai[apps]
```

> [!TIP]
> To switch from the stable release channel to the nightly release channel,
> see [`nightly_releases.md`](./nightly_releases.md).

### Test the installation.

```
python -m shortfin_apps.sd.server --help
```

## Getting started

As part of our current release we support serving [SDXL](https://stablediffusionxl.com/) and [Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) variants as well as an initial release of `amdsharktank`, amdshark's model development toolkit which is leveraged in order to compile these models to be high performant.

### SDXL

To get started with SDXL, please follow the [SDXL User Guide](../shortfin/python/shortfin_apps/sd/README.md#Start-SDXL-Server)


### Llama 3.1

To get started with Llama 3.1, please follow the [Llama User Guide][1].

* Once you've set up the Llama server in the guide above, we recommend that you use [SGLang Frontend](https://sgl-project.github.io/frontend/frontend.html) by following the [Using `shortfin` with `sglang` guide](shortfin/llm/user/shortfin_with_sglang_frontend_language.md)
* If you would like to deploy LLama on a Kubernetes cluster we also provide a simple set of instructions and deployment configuration to do so [here](shortfin/llm/user/llama_serving_on_kubernetes.md).
* Finally, if you'd like to leverage the instructions above to run against a different variant of Llama 3.1, it's supported. However, you will need to generate a gguf dataset for that variant (explained in the [user guide][1]). In future releases, we plan to streamline these instructions to make it easier for users to compile their own models from HuggingFace.

[1]: shortfin/llm/user/llama_serving.md
