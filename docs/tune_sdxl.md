# Steps to run tuning on SDXL models
- Precompile
- IREE build with tracy enabled
  - Baseline benchmarking
  - Run Tracy in parallel
- IREE compilation to generate dispatches
- Tuning


## Precompile the model
Create and run docker
For more details, refer instructions at [here](https://github.com/nod-ai/SHARK-MLPERF/tree/staging-v5.1/code/stable-diffusion-xl#amd-mlperf-inference-docker-container-setup)

Run precompilation script from docker (you should be at directory: `/mlperf/harness`):
```shell
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json --model_weights $PWD/new_weight
```

It will generate model and compile it at directory: `new_weight`
You should see files in directory `new_weight` something similar to following:
```shell
./bin/sdxl/stable_diffusion_xl_base_1_0_scheduled_unet_bs2_64_1024x1024_fp8_amdgpu-gfx942.vmfb
./bin/sdxl/stable_diffusion_xl_base_1_0_vae_bs1_1024x1024_fp16_amdgpu-gfx942.vmfb
./bin/sdxl/stable_diffusion_xl_base_1_0_clip_bs2_64_fp16_amdgpu-gfx942.vmfb
./genfiles/sdxl/stable_diffusion_xl_base_1_0_clip_dataset_fp16.irpa
./genfiles/sdxl/stable_diffusion_xl_base_1_0_vae_bs1_1024x1024_fp16.mlir
./genfiles/sdxl/stable_diffusion_xl_base_1_0_clip_bs2_64_fp16.mlir
./genfiles/sdxl/stable_diffusion_xl_base_1_0_scheduled_unet_bs2_64_1024x1024_fp8.mlir
./genfiles/sdxl/version.txt
./genfiles/sdxl/stable_diffusion_xl_base_1_0_vae_dataset_fp16.irpa
./genfiles/sdxl/stable_diffusion_xl_base_1_0_punet_dataset_i8.irpa
```

Exit from the docker now.


## iree build with tracy enabled

Build iree with tracy enabled
```shell
git clone  https://github.com/iree-org/iree.git
cd iree
git submodule update --init
python3.11 -m venv my_env
source my_
env/bin/activate
pip install numpy

cmake -G Ninja -B ../iree-build/ -S . \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DIREE_ENABLE_ASSERTIONS=ON \
   -DIREE_ENABLE_SPLIT_DWARF=ON \
   -DIREE_ENABLE_THIN_ARCHIVES=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DIREE_BUILD_PYTHON_BINDINGS=ON \
   -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON \
   -DIREE_ENABLE_LLD=ON \
   -DPYTHON3_EXECUTABLE="$(which python3)" \
   -DIREE_BUILD_TRACY=ON \
   -DIREE_ENABLE_RUNTIME_TRACING=ON


cmake --build ../iree-build/
```

Set environment
```shell
export PATH=<Dir Containing iree-build>/iree-build/tools/:$PATH
export PYTHONPATH=<Dir Containing iree-build>/iree-build/compiler/bindings/python:<Dir Containing iree-build>/iree-build/runtime/bindings/python
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
```

### Run tracy
From a different shell run tracy-capture tool. You can use `iree-tracy-capture` tool from above mentioned `iree` build or from nightly.
```shell
TRACY_PORT=8086 iree-tracy-capture -o out1.tracy -p 8086 -f
```


### Baseline benchmarking
Run following command using the files generated during the precompilation step.
Use tool from the above mentioned `iree` build which is tracy enabled.
```shell
IREE_PY_RUNTIME=tracy TRACY_PORT=8086 TRACY_NO_EXIT=1 iree-benchmark-module   --device=hip   --device_allocator=caching  \
--module=new_weight/bin/sdxl/stable_diffusion_xl_base_1_0_punet_bs16_64_1024x1024_fp8_amdgpu-gfx942.vmfb  \
--parameters="model=new_weight/genfiles/sdxl/stable_diffusion_xl_base_1_0_punet_dataset_i8.irpa" \
--function=run_forward   --input=16x4x128x128xf16 --input=1xsi64 --benchmark_repetitions=20
```

Here, following are the meaning of the inputs to the benchmark:
- `vmfb` file: compiled version of the MLIR file that you want to tune.
- `irpa` file: corresponding weights file.
- `function` option: The function which you want to target for tuning. Generally, it is the biggest function that will give good numbers after tuning.
- `input` option: From MLIR, you can find the input sizes for the function that you are targetting for tuning.

### Read tracy output
Upload generated tracy output file i.e. `out1.tracy` to an online tracy file reader: [link](tracy.nereid.pl)

### Find out dispatch for tuning
Indentify the dispatches either of `MatMul` or `Conv` which are taking time above than 1%. Note down, the dispatch number.

## IREE compilation to generate dispatches
```
iree-compile <IR Path> \
--iree-config-add-tuner-attributes \
--iree-hal-executable-debug-level=3 \
--iree-hal-dump-executable-files-to=dispatchOutput \
--iree-hal-target-backends=rocm \
--iree-hip-target=gfx942 \
--iree-execution-model=async-external \
--iree-global-opt-propagate-transposes=1 \
--iree-opt-const-eval=0 \
--iree-opt-outer-dim-concat=1 \
--iree-opt-aggressively-propagate-transposes=1 \
--iree-dispatch-creation-enable-aggressive-fusion \
--iree-hal-force-indirect-command-buffers \
--iree-llvmgpu-enable-prefetch=1 \
--iree-codegen-gpu-native-math-precision=1 \
--iree-opt-data-tiling=0 \
--iree-hal-memoization=1 \
--iree-opt-strip-assertions \
--iree-codegen-llvmgpu-early-tile-and-fuse-matmul=1 \
--iree-stream-resource-memory-model=discrete \
--iree-vm-target-truncate-unsupported-floats \
--iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental),iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})' \
--iree-dispatch-creation-enable-fuse-horizontal-contractions=0
```

`<IR_Path>`: Get MLIR generated from precompilation step

This will generate benchmark dispatch IR inside `dispatchOutput` directory


## Run tuner

### Setting up
Clone [shark-ai](https://github.com/nod-ai/shark-ai/)

Run commands:
```
cd shark-ai/tuner
pip install -r requirements-tuner.txt
pip install -r requirements-dev.txt
```

Create following files:
`compile_flags.txt`
```
--iree-hal-target-backends=rocm
--iree-hip-target=gfx942
--iree-vm-bytecode-module-output-format=flatbuffer-binary
--iree-dispatch-creation-enable-aggressive-fusion
--iree-dispatch-creation-enable-fuse-horizontal-contractions=false
--iree-opt-aggressively-propagate-transposes=true
--iree-codegen-llvmgpu-use-vector-distribution=true
--iree-opt-data-tiling=false
--iree-vm-target-truncate-unsupported-floats
--iree-opt-outer-dim-concat=true
--iree-codegen-gpu-native-math-precision=true
--iree-hal-indirect-command-buffers=true
--iree-stream-resource-memory-model=discrete
--iree-hal-memoization=true
--iree-opt-strip-assertions
--iree-global-opt-propagate-transposes=true
--iree-opt-const-eval=false
--iree-llvmgpu-enable-prefetch=true
--iree-execution-model=async-external
--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))
```

`model_benchmark_flags.txt`:
```
--device_allocator=caching
--parameters=model=<IRPA_FILE_PATH>
--function=run_forward
--input=16x4x128x128xf16
--input=32x64x2048xf16
--input=32x1280xf16
--input=32x6xf16
--input=1xf16
--input=1xsi64
--input=100xf32
```
Update IRPA_FILE_PATH with the `irpa` file generated from `iree-compilation` step.
Add input sizes as per the arguments of the function to be tuned.


### Command to run tuner

```
python -m examples.simple \
  <IR_PATH> \
  --simple-compile-flags-file=compile_flags.txt  \
  --simple-model-benchmark-flags-file=model_benchmark_flags.txt \
  --devices=hip://0 --num-candidates=9000  \
  --simple-num-dispatch-candidates=256 \
  --simple-num-model-candidates=200 \
  --codegen-pipeline=llvmgpu_tile_and_fuse \
  --no-reduce-shared-memory-bank-conflicts-options=True,False
```
`<IR_PATH>`: Provide same MLIR path given to iree-compile

You can change `num-candidates`, `simple-num-dispatch-candidates` or `simple-num-model-candidates` as per need.
It will take some time to finish tuner. You should see the final candidates spec that you can include in td_spec for `harness` run to get better SPS.


## Reference links:
[Tuner Example](https://github.com/amd-vivekag/shark-ai/blob/main/tuner/examples/simple/README.md)
