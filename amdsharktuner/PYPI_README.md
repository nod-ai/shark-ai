# amdshark Tuner
amdshark Tuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.

## Installation
Install amdshark Tuner from PyPI:
```shell
pip install amdsharktuner
```

This will install all required dependencies including IREE compiler and runtime.

You can use the latest nightly IREE python bindings:
```shell
pip install --upgrade --pre \
    iree-base-compiler \
    iree-base-runtime \
    --find-links https://iree.dev/pip-release-links.html
```

Verify that the environment is set up correctly:
```shell
python -m model_tuner --help
```

or

```shell
python -m dispatch_tuner --help
```

### Model Tuner
Use the Model Tuner to tune a dispatch and a model:
```shell
python -m model_tuner double_mmt.mlir mmt_benchmark.mlir \
    --compile-flags-file=compile_flags.txt \
    --model-benchmark-flags-file=model_benchmark_flags.txt \
    --devices=hip://0 \
    --num-candidates=30 \
    --model-tuner-num-dispatch-candidates=5 \
    --model-tuner-num-model-candidates=3`
```

Refer to [Mode Tuner README](https://github.com/nod-ai/amdshark-ai/tree/main/amdsharktuner/model_tuner) for detailed information on flags and MLIR files.

### Dispatch Tuner
Use the Dispatch Tuner to tune a dispatch:
```shell
python -m dispatch_tuner dispatch_sample.mlir dispatch_sample_benchmark.mlir \
    --compile-flags-file=compile_flags.txt
    --devices=hip://0 \
    --num-candidates=30
```

Refer to [Dispatch Tuner README](https://github.com/nod-ai/amdshark-ai/tree/main/amdsharktuner/dispatch_tuner) for detailed information on flags and MLIR files.
