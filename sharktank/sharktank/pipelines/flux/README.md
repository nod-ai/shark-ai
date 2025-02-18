# Flux.1 dynamo exports

### Quick Start

All the exports in this directory are done through `export.py`, with the CLI syntax as follows:
```shell
python -m sharktank.pipelines.flux.export_parameters --dtype <fp32/fp16/bf16> --input-dir <input-dir> --output-dir <output-dir>

python -m sharktank.pipelines.flux.export_components --model="flux-dev" --component=<clip/vae/t5xxl/mmdit/scheduler> --precision=<fp32/fp16/bf16>
```
