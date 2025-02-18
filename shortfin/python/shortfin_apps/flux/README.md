# Flux.1 Server and CLI

This directory contains a [Flux](https://blackforestlabs.ai/#get-flux) inference server, CLI and support components. More information about FLUX.1 on [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev).

## Install

For [nightly releases](../../../../docs/nightly_releases.md)
For our [stable release](../../../../docs/user_guide.md)

## Start Flux Server
The server will prepare runtime artifacts for you.

By default, the port is set to 8000. If you would like to change this, use `--port` in each of the following commands.

You can check if this (or any) port is in use on Linux with `ss -ntl | grep 8000`.

From a source checkout of shortfin:
```
python -m shortfin_apps.flux.server --model_config=./python/shortfin_apps/flux/examples/flux_dev_config_mixed.json --device=amdgpu --fibers_per_device=1 --workers_per_device=1 --isolation="per_fiber" --flagfile=./python/shortfin_apps/flux/examples/flux_flags_gfx942.txt --build_preference=precompiled
```
 - Wait until your server outputs:
```
INFO - Application startup complete.
INFO - Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
## Run the Flux Client

 - Run a CLI client in a separate shell:
```
python -m shortfin_apps.flux.simple_client --interactive
```
