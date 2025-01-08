# Simple Example Tuner

Example of tuning a dispatch and a full model.

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Running the Tuner

### Choose a model to tune
This example uses the simple `double_mmt.mlir` file.

### Generate a benchmark file
Use the usual `iree-compile` command for your model, add
`--iree-hal-dump-executable-files-to=dump --iree-config-add-tuner-attributes`,
and get the dispatch benchmark that you want to tune. For example:
```shell
mkdir tmp
iree-compile double_mmt.mlir --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 --iree-hal-dump-executable-files-to=tmp/dump \
    --iree-config-add-tuner-attributes -o /dev/null

cp tmp/dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir tmp/mmt_benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
cd ../../
python -m examples.simple examples/simple/double_mmt.mlir \
    examples/simple/tmp/mmt_benchmark.mlir \
    --devices=hip://0 --num-candidates=30 \
    --simple-num-dispatch-candidates=5 --simple-num-model-candidates=3 \
```

### Basic Usage
```shell
python -m examples.simple <model_file_path> <benchmark_file_path> \
    --devices=hip://0 --num-candidates=1024 \
    --test-num-dispatch-candidates=<num_dispatch_candidates> \
    --test-num-model-candidates=<num_model_candidates> \
    --test-hip-target=<hip_target> \
    --num-candidates=<num_generated_candidates> \
    --codegen-pipeline=<codegen_pipeline>
```
