"""Flux text-to-image generation pipeline."""

import argparse
import math
from os import PathLike
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
from transformers import CLIPTokenizer

from sharktank.layers.base import BaseLayer
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.models.punet.model import Unet2DConditionModel
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.types import Dataset
from .scheduler import SchedulingModel


class SdxlPipeline(BaseLayer):
    """Pipeline for text-to-image generation using the SDXL model."""

    def __init__(
        self,
        clip_path: PathLike,
        unet_path: PathLike,
        vae_path: PathLike,
        clip_tokenizer_path: Optional[PathLike] = None,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        base_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        default_num_inference_steps: Optional[int] = None,
    ):
        """Initialize the Flux pipeline."""
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.get_default_device()
        self.dtype = dtype
        if not default_num_inference_steps:
                self.default_num_inference_steps = 20

        if clip_tokenizer_path:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(base_model_name)

        # Load CLIP
        clip_dataset = Dataset.load(clip_path)
        clip_config = ClipTextConfig.from_properties(clip_dataset.properties)
        self.clip_model = ClipTextModel(
            theta=clip_dataset.root_theta, config=clip_config
        )
        self.add_module("clip_model", self.clip_model)
        self.clip_model.to(device)

        # Load SDXL Unet
        self.unet_model = Unet2DConditionModel.from_irpa(
            unet_dataset
        )
        self.add_module("unet_model", self.unet_model)
        self.unet_model.to(device)

        # Load VAE
        vae_dataset = Dataset.load(vae_path)
        self.vae_model = VaeDecoderModel.from_dataset(vae_dataset)
        self.add_module("vae_model", self.vae_model)
        self.vae_model.to(device)

        self._rng = torch.Generator(device="cpu")

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        latents: Optional[Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Tensor:
        """Generate images from a prompt

        Args:
            prompt: Text prompt for image generation
            height: Height of output image
            width: Width of output image
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            seed: Random seed for reproducibility

        Returns:
            Image tensor

        """
        if not self.t5_tokenizer or not self.clip_tokenizer:
            raise ValueError("Tokenizers must be provided to use the __call__ method")

        clip_prompt_ids = self.tokenize_prompt(prompt)
        if not latents:
            latents = self.transformer_model._get_noise(
                1,
                height,
                width,
                seed=seed,
            )

        return self.forward(
            t5_prompt_ids,
            clip_prompt_ids,
            latents,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def forward(
        self,
        t5_prompt_ids: Tensor,
        clip_prompt_ids: Tensor,
        latents: Tensor,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Tensor:
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        # Adjust dimensions to be multiples of 16
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        # Generate initial noise
        x = latents

        # Prepare inputs
        inp = self._prepare(
            self.t5_model, self.clip_model, t5_prompt_ids, clip_prompt_ids, x
        )
        timesteps = self._get_schedule(
            num_inference_steps, inp["img"].shape[1], shift=True
        )

        # Denoise
        x = self._denoise(
            **inp,
            num_steps=num_inference_steps,
            guidance=guidance_scale,
        )

        # Decode latents
        x = self._unpack(x.to(dtype=self.dtype), height, width)
        x = self.ae_model(x)

        x = x[0]
        x = x.cpu()
        x = x.clamp(-1, 1)
        x = rearrange(x, "c h w -> h w c")
        return x.float()

    def _prepare(
        self,
        t5: T5Encoder,
        clip: ClipTextModel,
        t5_prompt_ids: Tensor,
        clip_prompt_ids: Tensor,
        img: Tensor,
    ) -> dict[str, Tensor]:
        """Prepare inputs for the unet model.

        Args:
            clip: CLIP model for text encoding
            clip_prompt_ids: Tokenized CLIP prompt IDs
            img: Initial noise tensor

        Returns:
            Dictionary containing prepared inputs for unet.
        """
        # Process text through CLIP
        pe, te = clip(clip_prompt_ids)

        # Return prepared inputs
        return {
            "img": img,
            "pe": prompt_embeds,
            "te": text_embeds,
        }

    def _denoise(
        self,
        # model input
        img: Tensor,
        pe: Tensor,
        te: Tensor,
        vec: Tensor,
        # sampling parameters
        num_steps: int,
        guidance: float = 7.5,
        # extra img tokens
        img_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Denoise the latents through the diffusion process."""
        img, t_ids, timesteps, sigmas = self.scheduler.initialize(img, num_steps)
        
        for step in range(timesteps):
            latents, t, sigma, next_sigma = self.scheduler.scale_model_input(img, step, timesteps, sigmas)
            noise_pred = self.unet_model(
                latents,
                t,
                pe,
                te,
                t_ids,
                guidance
            )
            img = self.scheduler.step(noise_pred, img, sigma, next_sigma)

        return img

    def tokenize_prompt(self, prompt: str, neg_prompt: str) -> tuple[Tensor, Tensor]:
        """Tokenize a prompt using CLIP tokenizers.

        Args:
            prompt: Text prompt to tokenize
            neg_prompt: Text negative prompt to tokenize.

        Returns:
            A list of clip token arrays [positive, negative, positive pooled, negative pooled]
        """
        # CLIP tokenization
        input_ids_list = []
        neg_ids_list = []
        for tokenizer in self.tokenizers:
            input_ids = tokenizer.encode(prompt).input_ids
            input_ids_list.append(input_ids)
            neg_ids = tokenizer.encode(neg_prompt).input_ids
            neg_ids_list.append(neg_ids)
        ids_list = [*input_ids_list, *neg_ids_list]
        clip_prompt_ids = [torch.tensor(token, dtype=torch.long) for token in ids_list]
        return clip_prompt_ids


def main():
    """Example usage of FluxPipeline."""
    parser = argparse.ArgumentParser(
        description="Flux text-to-image generation pipeline"
    )

    # Model paths
    parser.add_argument(
        "--clip-path",
        required=True,
        help="Path to CLIP model",
    )
    parser.add_argument(
        "--unet-path",
        default="/data/flux/FLUX.1-dev/transformer/model.irpa",
        help="Path to Transformer model",
    )
    parser.add_argument(
        "--vae-path",
        default="/data/flux/FLUX.1-dev/vae/model.irpa",
        help="Path to VAE model",
    )

    # Generation parameters
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of output image"
    )
    parser.add_argument("--width", type=int, default=1024, help="Width of output image")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Scale for classifier-free guidance",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Other parameters
    parser.add_argument(
        "--prompt",
        default="a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
        help="Text prompt for image generation",
    )
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model",
    )

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Initialize pipeline
    pipeline = SdxlPipeline(
        clip_path=args.clip_path,
        unet_path=args.unet_path,
        vae_path=args.vae_path,
        dtype=dtype_map[args.dtype],
    )

    # Generate image
    x = pipeline(
        prompt=args.prompt,
        neg_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # Transform and save first image
    image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    image.save(args.output, quality=95, subsampling=0)


if __name__ == "__main__":
    main()
