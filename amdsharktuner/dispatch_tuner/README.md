# Getting Started with Dispatch Tuner

Example of tuning a dispatch using `dispatch_tuner`

## Environments
Follow instructions in [`/amdsharktuner/README.md`](../README.md)

## Running the Tuner

### Choose a dispatch to tune
This example uses the simple `dispatch_sample.mlir` file.

### Generate a benchmark file
Use the usual `iree-compile` command for problem dispatch, add
`--iree-hal-dump-executable-files-to=dump --iree-config-add-tuner-attributes`,
and get the dispatch benchmark that you want to tune. For example:

```shell
iree-compile dispatch_sample.mlir --iree-hal-target-device=hip \
    --iree-hip-target=gfx942 --iree-hal-dump-executable-files-to=tmp/dump \
    --iree-config-add-tuner-attributes -o /dev/null

cp tmp/dump/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir tmp/dispatch_sample_benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use following command:

```shell
cd amdshark-ai/amdsharktuner
python -m dispatch_tuner dispatch_tuner/dispatch_sample.mlir \
    dispatch_tuner/tmp/dispatch_sample_benchmark.mlir \
    --compile-flags-file=dispatch_tuner/compile_flags.txt \
    --devices=hip://0 --num-candidates=30
```

> Example input format for multiple devices: use a comma-separated list, such as `--devices=hip://0,hip://1`


[!TIP]
Use the `--starter-td-spec` option to pass an existing td spec for the run.
You can use following default td spec: [Default Spec](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir).

## Algorithm of the dispatch tuner
### Tuning algorithm
1. Generate Candidate specs
2. Compile candidate
3. Benchmark for candidates
    - Baseline benchmark for candidates (now serially over all the given devices)
    - Candidate benchmark (parallel over all the given devices)
    - Second baseline run to check for any regression
    - Return top candidates
