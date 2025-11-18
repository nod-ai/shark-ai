"""Flux text-to-image generation pipeline."""

import argparse
import math
from os import PathLike
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
from transformers import CLIPTokenizer, T5Tokenizer

from amdsharktank.layers.base import BaseLayer
from amdsharktank.models.t5 import T5Config, T5Encoder
from amdsharktank.models.clip import ClipTextModel, ClipTextConfig
from amdsharktank.models.flux.flux import FluxModelV1, FluxParams
from amdsharktank.models.vae.model import VaeDecoderModel
from amdsharktank.types import Dataset
from amdsharktank import ops


class FluxPipeline(BaseLayer):
    """Pipeline for text-to-image generation using the Flux model."""

    def __init__(
        self,
        t5_path: PathLike,
        clip_path: PathLike,
        transformer_path: PathLike,
        ae_path: PathLike,
        t5_tokenizer_path: Optional[PathLike] = None,
        clip_tokenizer_path: Optional[PathLike] = None,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        base_model_name: str = "flux_dev",
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
            if base_model_name in ["dev", "flux_dev"]:
                self.default_num_inference_steps = 50
            else:
                self.default_num_inference_steps = 4

        if t5_tokenizer_path:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
        if clip_tokenizer_path:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path)

        # Load T5
        t5_dataset = Dataset.load(t5_path)
        t5_config = T5Config.from_gguf_properties(
            t5_dataset.properties,
            feed_forward_proj="gated-gelu",
        )
        self.t5_model = T5Encoder(theta=t5_dataset.root_theta, config=t5_config)
        self.add_module("t5_model", self.t5_model)
        self.t5_model.to(device)

        # Load CLIP
        clip_dataset = Dataset.load(clip_path)
        clip_config = ClipTextConfig.from_properties(clip_dataset.properties)
        self.clip_model = ClipTextModel(
            theta=clip_dataset.root_theta, config=clip_config
        )
        self.add_module("clip_model", self.clip_model)
        self.clip_model.to(device)

        # Load Flux Transformer
        transformer_dataset = Dataset.load(transformer_path)
        transformer_params = FluxParams.from_hugging_face_properties(
            transformer_dataset.properties
        )
        self.transformer_model = FluxModelV1(
            theta=transformer_dataset.root_theta, params=transformer_params
        )
        self.add_module("transformer_model", self.transformer_model)
        self.transformer_model.to(device)

        # Load VAE
        ae_dataset = Dataset.load(ae_path)
        self.ae_model = VaeDecoderModel.from_dataset(ae_dataset)
        self.add_module("ae_model", self.ae_model)
        self.ae_model.to(device)

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

        Raises:
            ValueError: If tokenizers are not provided
        """
        if not self.t5_tokenizer or not self.clip_tokenizer:
            raise ValueError("Tokenizers must be provided to use the __call__ method")

        t5_prompt_ids, clip_prompt_ids = self.tokenize_prompt(prompt)
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
            timesteps=timesteps,
            guidance=guidance_scale,
        )

        # Decode latents
        x = self._unpack(x.to(dtype=self.dtype), height, width)
        x = self.ae_model(x)

        x = x[0]
        x = x.cpu()
        x = x.clamp(-1, 1)
        x = rearrange(x, "c h w -> h w c")
        return x.to(dtype=torch.float32)

    def _prepare(
        self,
        t5: T5Encoder,
        clip: ClipTextModel,
        t5_prompt_ids: Tensor,
        clip_prompt_ids: Tensor,
        img: Tensor,
    ) -> dict[str, Tensor]:
        """Prepare inputs for the transformer model.

        Args:
            t5: T5 model for text encoding
            clip: CLIP model for text encoding
            t5_prompt_ids: Tokenized T5 prompt IDs
            clip_prompt_ids: Tokenized CLIP prompt IDs
            img: Initial noise tensor

        Returns:
            Dictionary containing prepared inputs for the transformer
        """
        bs, c, h, w = img.shape

        # Prepare image and position IDs
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        # Get text embeddings
        # Process text through T5
        txt = t5(t5_prompt_ids)["last_hidden_state"]
        txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device)

        # Process text through CLIP
        vec = clip(clip_prompt_ids)["pooler_output"]

        # Return prepared inputs
        return {
            "img": img,
            "img_ids": img_ids,
            "txt": txt,
            "txt_ids": txt_ids,
            "vec": vec,
        }

    def _time_shift(self, mu: float, sigma: float, t: Tensor) -> Tensor:
        """Apply time shift to the schedule."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        """Get linear interpolation function."""
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def _get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        """Generate sampling schedule.

        Args:
            num_steps: Number of diffusion steps
            image_seq_len: Length of the image sequence
            base_shift: Base shift value for schedule adjustment
            max_shift: Maximum shift value for schedule adjustment
            shift: Whether to apply schedule shifting

        Returns:
            List of timesteps for the diffusion process
        """
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # estimate mu based on linear estimation between two points
            mu = self._get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = self._time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()

    def _denoise(
        self,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 3.5,
        # extra img tokens
        img_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Denoise the latents through the diffusion process."""
        # this is ignored for schnell
        guidance_vec = ops.full(
            (img.shape[0],), guidance, device=img.device, dtype=self.dtype
        )
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_curr_vec = ops.full(
                (img.shape[0],), t_curr, dtype=self.dtype, device=img.device
            )
            t_prev_vec = ops.full(
                (img.shape[0],), t_prev, dtype=self.dtype, device=img.device
            )
            pred = self.transformer_model(
                img=ops.cat((img, img_cond), dim=-1) if img_cond is not None else img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_curr_vec,
                guidance=guidance_vec,
            )
            print(t_prev, t_curr)
            img = img + (t_prev_vec - t_curr_vec) * pred

        return img

    def _unpack(self, x: Tensor, height: int, width: int) -> Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    def tokenize_prompt(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Tokenize a prompt using T5 and CLIP tokenizers.

        Args:
            prompt: Text prompt to tokenize

        Returns:
            Tuple of (t5_prompt_ids, clip_prompt_ids) tensors
        """
        # T5 tokenization
        t5_prompt_ids = [
            self.t5_tokenizer(
                p,
                truncation=True,
                max_length=self.max_length,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ).input_ids
            for p in [prompt]
        ]
        t5_prompt_ids = torch.tensor(t5_prompt_ids, dtype=torch.long)

        # CLIP tokenization
        clip_prompt_ids = [
            self.clip_tokenizer(
                p,
                truncation=True,
                max_length=self.max_length,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ).input_ids
            for p in [prompt]
        ]
        clip_prompt_ids = torch.tensor(clip_prompt_ids, dtype=torch.long)

        return t5_prompt_ids, clip_prompt_ids


def main():
    """Example usage of FluxPipeline."""
    parser = argparse.ArgumentParser(
        description="Flux text-to-image generation pipeline"
    )

    # Model paths
    parser.add_argument(
        "--t5-path", default="/data/t5-v1_1-xxl/model.gguf", help="Path to T5 model"
    )
    parser.add_argument(
        "--clip-path",
        default="/data/flux/FLUX.1-dev/text_encoder/model.irpa",
        help="Path to CLIP model",
    )
    parser.add_argument(
        "--transformer-path",
        default="/data/flux/FLUX.1-dev/transformer/model.irpa",
        help="Path to Transformer model",
    )
    parser.add_argument(
        "--ae-path",
        default="/data/flux/FLUX.1-dev/vae/model.irpa",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--t5-tokenizer-path",
        default="/data/flux/FLUX.1-dev/tokenizer_2/",
        help="Path to T5 tokenizer",
    )
    parser.add_argument(
        "--clip-tokenizer-path",
        default="/data/flux/FLUX.1-dev/tokenizer/",
        help="Path to CLIP tokenizer",
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
        default="a photo of a forest with mist swirling around the tree trunks. "
        'The word "FLUX" is painted over it in big, red brush strokes with visible texture',
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
    pipeline = FluxPipeline(
        t5_path=args.t5_path,
        clip_path=args.clip_path,
        transformer_path=args.transformer_path,
        ae_path=args.ae_path,
        t5_tokenizer_path=args.t5_tokenizer_path,
        clip_tokenizer_path=args.clip_tokenizer_path,
        dtype=dtype_map[args.dtype],
    )

    # Generate image
    x = pipeline(
        prompt=args.prompt,
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
