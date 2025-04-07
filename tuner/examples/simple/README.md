# Simple Example Tuner

Example of tuning a dispatch and a full model.

## Environments
Follow instructions in [`/tuner/README.md`](../../README.md)

## Running the Tuner

### Choose a model to tune
This example uses the simple `double_mmt.mlir` file.

### Generate a benchmark file
Use the usual `iree-compile` command for your model, add
`--iree-hal-dump-executable-files-to=dump --iree-config-add-tuner-attributes`,
and get the dispatch benchmark that you want to tune. For example:

```shell
mkdir tmp
iree-compile double_mmt.mlir --iree-hal-target-device=hip \
    --iree-hip-target=gfx942 --iree-hal-dump-executable-files-to=tmp/dump \
    --iree-config-add-tuner-attributes -o /dev/null

cp tmp/dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir tmp/mmt_benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use following command:

```shell
cd ../../
python -m examples.simple examples/simple/double_mmt.mlir \
    examples/simple/tmp/mmt_benchmark.mlir \
    --simple-compile-flags-file=examples/simple/compile_flags.txt \
    --simple-model-benchmark-flags-file=examples/simple/model_benchmark_flags.txt \
    --devices=hip://0 --num-candidates=30 \
    --simple-num-dispatch-candidates=5 --simple-num-model-candidates=3 \
```

[!TIP]
Use the `--starter-td-spec` option to pass an existing td spec for the run.
You can use following default td spec: [Default Spec](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir).

### Basic Usage

```shell
python -m examples.simple <model_file_path> <benchmark_file_path> \
    --devices=hip://0,hip://1 --num-candidates=1024 \
    --simple-compile-flags-file=<compile_flags_path> \
    --simple-model-benchmark-flags-file=<model_benchmark_flags_path> \
    --simple-num-dispatch-candidates=<num_dispatch_candidates> \
    --simple-num-model-candidates=<num_model_candidates> \
    --num-candidates=<num_generated_candidates> \
    --codegen-pipeline=<codegen_pipeline>
```
