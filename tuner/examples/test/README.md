# Example Tuner Test

Example of tuning a dispatch and full model.

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
iree-compile double_mmt.mlir --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 --iree-hal-dump-executable-files-to=dump \
    --iree-config-add-tuner-attributes -o /dev/null

cp dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir mmt_benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
python -m examples.test double_mmt.mlir mmt_benchmark.mlir \
    --test_num_dispatch_candidates=5 --test_num_model_candidates=3 \
    --num-candidates=30
```

### Basic Usage
```shell
python -m examples.test <model_file_path> <benchmark_file_path> \
    --test_num_dispatch_candidates=<num_dispatch_candidates> \
    --test_num_model_candidates=<num_model_candidates> \
    --test_hip_target=<hip_target> \
    --num-candidates=<num_generated_candidates> \
    --codegen-pipeline=<codegen_pipeline>
```
