# Flux.1 Server and CLI

This directory contains a [Flux](https://blackforestlabs.ai/#get-flux) inference server, CLI and support components. More information about FLUX.1 on [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev).

## Install

To get your environment ready, follow the [developer guide](https://github.com/nod-ai/shark-ai/blob/main/docs/developer_guide.md)

## Prepare artifacts

The flux model weights are gated. Once you have access, clone the huggingface repository (any of the following):
 - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
 - [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)


Steps to follow to export flux data for further steps:
1. Get access token from the [link](https://huggingface.co/settings/tokens/) and login using it: `huggingface-cli login`
2. Download the required repo:
  - For Flux-dev: `huggingface-cli download black-forest-labs/FLUX.1-dev`
  - For Flux-schnell: `huggingface-cli download black-forest-labs/FLUX.1-schnell`
3. Once downloaded, run the following command:
```
./sharktank/sharktank/pipelines/flux/export_from_hf.sh <flux_snapshot_path> <flux_dev | flux_schnell>
```

Here, `flux_snapshot_path` will be the path where snapshot got downloaded in Step 2. The path will look like this:
 - For Flux-dev: `/home/<user>/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/`
 - For Flux-schnell: `/home/<user>/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/`


## Start Flux Server
The server will prepare runtime artifacts for you.

By default, the port is set to 8000. If you would like to change this, use `--port` in each of the following commands.

You can check if this (or any) port is in use on Linux with `ss -ntl | grep 8000`.

The first time you run the server, you may need to wait for artifacts to download.

From a source checkout of shortfin, you must run from the `/shortfin` directory:
```
python -m shortfin_apps.flux.server --model_config=<config_path> --device=hip --fibers_per_device=1 --workers_per_device=1 --isolation="per_fiber" --build_preference=precompiled

```

Here, `config_path` will be:
 - For Flux-dev: `./python/shortfin_apps/flux/examples/flux_dev_config.json`
 - For Flux-schnell: `./python/shortfin_apps/flux/examples/flux_schnell_config.json`

Now, wait until your server outputs:
```
INFO - Application startup complete.
INFO - Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> [!NOTE]
> The option `--build_preference=precompiled` will download precompiled vmfbs to following directory: `~/.cache/shark/genfiles/flux/` and server will use that.
> If you want to compile your own vmfb, then use option: `--build_preference=compile`

## Run the Flux Client

 - Run a CLI client in a separate shell:
```
python -m shortfin_apps.flux.simple_client --interactive
```

Enter your prompts. The generated images will be stored at `./gen_imgs/`!

### Update flags

Please see --help for both the server and client for usage instructions.
