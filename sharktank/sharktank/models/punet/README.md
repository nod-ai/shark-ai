# punet - Partitioned UNet in the style used by SDXL.

This is an experimental, simplified variant of the `Unet2DConditionModel` from
diffusers. It can be run/compiled off of a `config.json` and `safetensors` files
from the `stable-diffusion-xl-base-1.0/unet` model.

It was originally written to experiment with sharding strategies (and the `p`
was for partitioned). It is now being used for quantization and other serving
optimizations as well.

It is pronounced like "pu-nay" or any other deriviation that matches the
mood of the user.

See the [huggingface repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet)
for parameters and config.

## Preparing dataset

If not sharding or quantizing the model, the official model can be imported
as is from huggingface:

```
model_dir=$(huggingface-cli download \
    stabilityai/stable-diffusion-xl-base-1.0 \
    unet/config.json unet/diffusion_pytorch_model.fp16.safetensors)
python -m amdsharktank.amdsharktank.tools.import_hf_dataset \
    --config-json $model_dir/unet/config.json \
    --output-irpa-file ~/models/punet_fp16.irpa \
    --params $model_dir/unet/diffusion_pytorch_model.fp16.safetensors
```

## Running reference model

```
python -m amdsharktank.models.punet.tools.run_diffuser_ref
```

## Run punet model

```
python -m amdsharktank.models.punet.tools.run_punet --irpa-file ~/models/punet_fp16.irpa
```

## Integration Testing

Integration testing is set up via pytest:

```
pytest -v amdsharktank/ -m model_punet
```

These perform a variety of expensive tests that involve downloading live data
that can be of considerable size. It is often helpful to run specific tests
with the `-s` option (stream output) and by setting `amdsharkTANK_TEST_ASSETS_DIR`
to an explicit temp directory (in this mode, the temp directory will not
be cleared, allowing you to inspect assets and intermediates -- but delete
manually as every run will accumulate). Filtering by test name with
`-k test_some_name` is also useful. Names have been chosen to facilitate this.

## Model Breaking Changes

If the format of the model or quantization parameters changes, then an update
must be coordinated. We are presently storing assets here:

https://huggingface.co/amd-amdshark/sdxl-quant-models/tree/main

The general procedure is:

* Create a branch in amdsharktank and in that repository.
* Make source changes to amdsharktank in the branch.
* Commit updates params.safetensors/config.json/quant_params.json to the
  branch in the sdxl-quant-models HF repo.
* Update the commit hash in `amdsharktank/integration/models/punet/integration_test.py`
  appropriately.
* Run the integration test with `amdsharkTANK_TEST_ASSETS_DIR=SOMEDIR`.
* Copy built assets from the test dir to the sdxl-quant-models/unet/int8/export
  path on the branch.
* Commit branches in both projects.

## License

Significant portions of this implementation were derived from diffusers,
licensed under Apache2: https://github.com/huggingface/diffusers
While much was a simple reverse engineering of the config.json and parameters,
code was taken where appropriate.
