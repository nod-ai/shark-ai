# SDXL torch exports

This directory contains export tooling for various functionalities extending from the SDXL1.0 inference pipeline.

To acquire MLIR export of a denoising module that combines EulerDiscreteScheduler with unet:
```
python export.py --component=scheduled_unet
```
To acquire VAE encode/decode module MLIR:
```
python export.py --component=vae
```
