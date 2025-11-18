# Getting Started with Model Tuner

Example of tuning a dispatch and a full model using `model_tuner`

## Environments
Follow instructions in [`/amdsharktuner/README.md`](../README.md)

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
cd amdshark-ai/amdsharktuner
python -m model_tuner model_tuner/double_mmt.mlir model_tuner/tmp/mmt_benchmark.mlir \
    --compile-flags-file=model_tuner/compile_flags.txt \
    --model-benchmark-flags-file=model_tuner/model_benchmark_flags.txt \
    --devices=hip://0 --num-candidates=30 \
    --model-tuner-num-dispatch-candidates=5 --model-tuner-num-model-candidates=3
```

> Example with multiple devices: `--devices=hip://0,--devices=hip://1`

[!TIP]
Use the `--starter-td-spec` option to pass an existing td spec for the run.
You can use following default td spec: [Default Spec](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir).

### Basic Usage

```shell
python -m model_tuner <model_file_path> <benchmark_file_path> \
    --devices=hip://0,hip://1 \
    --num-candidates=<max_candidate_num> \
    --compile-flags-file=<compile_flags_path> \
    --model-benchmark-flags-file=<model_benchmark_flags_path> \
    --num-dispatch-candidates=<num_dispatch_candidates> \
    --num-model-candidates=<num_model_candidates>
```
> Example input format for multiple devices: use a comma-separated list, such as `--devices=hip://0,hip://1`

## Algorithm of the model tuner
### Tuning algorithm
1. Generate Candidate specs
2. Compile candidate
3. Benchmark for candidates
    - Baseline benchmark for candidates (now serially over all the given devices)
    - Candidate benchmark (parallel over all the given devices)
    - Second baseline run to check for any regression
    - Return top candidates
4. Compile models
5. Benchmark for models
    - Baseline benchmark for models (now serially over all the given devices)
    - Model benchmark (parallel over all the given devices)
    - Second baseline run to check for any regression
    - Return top model candidates
