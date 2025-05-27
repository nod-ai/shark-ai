# SHARK artifacts export and compile for Wan2.1 text+audio to video benchmarks

## Getting MLIR

Currently, different torch versions are required for different exports. This is mostly due to VAE using torch.aten.as_strided in latest torch versions, which we don't have a torch to linalg lowering for -- we use latest torch for CLIP export (uses newer layer slicing not available in 2.5.1) and 2.5.1 for VAE export.

First, check out this branch (wan_exports) in your local repository clone.

### CLIP
Run:
```shell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
python export.py --force_export=clip
```
### VAE
Run:
```shell
pip uninstall torch torchvision torchaudio -y
pip install -r <shark_ai_root>/pytorch-rocm-requirements.txt
python export.py --force_export=vae
```
## Compile

### T5
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-execution-model=async-external  wan2_1_umt5xxl.mlir -o wan2_1_umt5xxl_gfx942.vmfb

### CLIP
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-execution-model=async-external --iree-global-opt-propagate-transposes=1 --iree-opt-const-eval=0 --iree-opt-outer-dim-concat=1 --iree-opt-aggressively-propagate-transposes=1 --iree-dispatch-creation-enable-aggressive-fusion --iree-hal-force-indirect-command-buffers --iree-llvmgpu-enable-prefetch=1 --iree-opt-data-tiling=0 --iree-hal-memoization=1 --iree-opt-strip-assertions --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=1 --iree-stream-resource-memory-model=discrete --iree-vm-target-truncate-unsupported-floats --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental),iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})' --iree-dispatch-creation-enable-fuse-horizontal-contractions=0 wan2_1_clip_512x512.mlir -o wan2_1_clip_512x512_gfx942.vmfb

### VAE
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-execution-model=async-external --iree-dispatch-creation-enable-fuse-horizontal-contractions=0  --iree-flow-inline-constants-max-byte-length=16 --iree-global-opt-propagate-transposes=1 --iree-opt-const-eval=0 --iree-opt-outer-dim-concat=1 --iree-opt-aggressively-propagate-transposes=1 --iree-dispatch-creation-enable-aggressive-fusion --iree-hal-force-indirect-command-buffers --iree-llvmgpu-enable-prefetch=1 --iree-opt-data-tiling=0 --iree-hal-memoization=1 --iree-opt-strip-assertions --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=1 --iree-stream-resource-memory-model=discrete --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental),iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})' wan2_1_vae_512x512.mlir -o wan2_1_vae_512x512_gfx942.vmfb 

iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-execution-model=async-external  wan2_1_vae_512x512.mlir -o wan2_1_vae_512x512_gfx942.vmfb

## Run validation
### T5
HIP_VISIBLE_DEVICES=1 iree-run-module --module=wan2_1_umt5xxl_gfx942.vmfb --input=@umt5xxl_input.npy --function=main --device=hip://0 --parameters=model=wan2_1_umt5xxl_bf16.irpa &> t5_out.txt

### CLIP
### VAE
HIP_VISIBLE_DEVICES=1 iree-run-module --module=wan2_1_vae_512x512_gfx942.vmfb --input=@vae_encode_input.npy --function=encode --device=hip://0 --parameters=model=wan2_1_vae_bf16.irpa &> vae_out.txt

HIP_VISIBLE_DEVICES=1 iree-run-module --module=wan2_1_vae_512x512_cpu.vmfb --input=@vae_encode_input.npy --function=encode --device=local-task --parameters=model=wan2_1_vae_bf16.irpa &> vae_out.txt

## Run model benchmarks
### CLIP
iree-benchmark-module --module=wan2_1_clip_512x512_gfx942.vmfb --input=@clip_input1.npy --input=@clip_input2.npy --function=forward --device=hip://0 --parameters=model=wan2_1_clip_rand_bf16.irpa

### VAE
iree-benchmark-module --module=wan2_1_vae_512x512_gfx942.vmfb --input=@vae_encode_input.npy --function=encode --device=hip://0 --parameters=model=wan2_1_vae_bf16.irpa

iree-benchmark-module --module=wan2_1_vae_512x512_gfx942.vmfb --input=@vae_decode_input.npy --function=decode --device=hip://0 --parameters=model=wan2_1_vae_bf16.irpa

### Tracing
Add `TRACY_PORT=8091 HIP_VISIBLE_DEVICES=1 IREE_PY_RUNTIME=tracy` and run `iree-tracy-capture -p 8091 -o my_module_trace.tracy`
