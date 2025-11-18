# VAE decoder

This is vae implemented in the style used for SDXL and referenced from diffusers implementation.

## Preparing dataset
If not sharding or quantizing, the official model can be imported as from huggingface:

```
model_dir=$(huggingface-cli download \
    stabilityai/stable-diffusion-xl-base-1.0 \
    vae/config.json vae/diffusion_pytorch_model.safetensors)

python -m amdsharktank.models.punet.tools.import_hf_dataset \
    --params $model_dir/vae/diffusion_pytorch_model.safetensors
    --config-json $model_dir/vae/config.json --output-irpa-file ~/models/vae.irpa
```

# Run Vae decoder model eager mode
# Sample SDXL command
```
python -m amdsharktank.models.vae.tools.run_vae --irpa-file ~/models/vae.irpa --device cpu --dtype=float32
```
# Sample Flux command to run through iree and compare vs huggingface diffusers torch model
```
python -m amdsharktank.models.vae.tools.run_vae --irpa-file ~/models/vae.irpa --device cpu --compare_vs_torch --dtype=float32 --amdsharktank_config=flux --torch_model=black-forest-labs/FLUX.1-dev
```
## License

Significant portions of this implementation were derived from diffusers,
licensed under Apache2: https://github.com/huggingface/diffusers
While much was a simple reverse engineering of the config.json and parameters,
code was taken where appropriate.
