import os
import math
from typing import Callable

import torch
from einops import repeat, rearrange
from diffusers import FluxTransformer2DModel

from iree.turbine.aot import *


def get_local_path(local_dir, model_dir):
    model_local_dir = os.path.join(local_dir, model_dir)
    if not os.path.exists(model_local_dir):
        os.makedirs(model_local_dir)
    return model_local_dir


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps


class FluxScheduler(torch.nn.Module):
    def __init__(self, max_length, torch_dtype, is_schnell=False):
        super().__init__()
        self.is_schnell = is_schnell
        self.max_length = max_length
        timesteps = [torch.empty((100), dtype=torch_dtype, requires_grad=False)] * 100
        for i in range(1, 100):
            schedule = get_schedule(i, max_length, shift=not self.is_schnell)
            timesteps[i] = torch.nn.functional.pad(schedule, (0, 99 - i), "constant", 0)
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()

    def prepare(self, num_steps):
        timesteps = self.timesteps[num_steps]
        return timesteps


class FluxModelCFG(torch.nn.Module):
    def __init__(
        self,
        torch_dtype,
        model_id="flux-dev",
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.mmdit = FluxTransformer2DModel.from_single_file(
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
        ).to(torch_dtype)
        self.batch_size = batch_size * 2
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = rearrange(img_ids, "h w c -> (h w) c")
        self.txt_ids = torch.zeros(max_length, 3)

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_prev = torch.index_select(timesteps, 0, step + 1)
        t_vec = t_curr.repeat(self.batch_size)

        pred = self.mmdit(
            hidden_states=img,
            img_ids=self.img_ids,
            encoder_hidden_states=txt,
            txt_ids=self.txt_ids,
            pooled_projections=vec,
            timestep=t_vec,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        pred_uncond, pred = torch.chunk(pred, 2, dim=0)
        pred = pred_uncond + guidance_scale * (pred - pred_uncond)
        img = img + (t_prev - t_curr) * pred
        return img


class FluxModelSchnell(torch.nn.Module):
    def __init__(
        self,
        torch_dtype,
        model_id="flux-schnell",
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        if "schnell" in model_id:
            self.mmdit = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors"
            ).to(torch_dtype)
        elif "dev" in model_id:
            self.mmdit = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
            ).to(torch_dtype)
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = repeat(img_ids, "h w c -> (h w) c")
        self.txt_ids = torch.zeros(max_length, 3)

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_prev = torch.index_select(timesteps, 0, step + 1)
        t_vec = t_curr.repeat(self.batch_size)

        pred = self.mmdit(
            hidden_states=img,
            img_ids=self.img_ids,
            encoder_hidden_states=txt,
            txt_ids=self.txt_ids,
            pooled_projections=vec,
            timestep=t_vec,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        img = img + (t_prev - t_curr) * pred
        return img


@torch.no_grad()
def get_flux_sampler_model(
    local_dir,
    hf_model_path,
    img_height=1024,
    img_width=1024,
    compression_factor=8,
    max_len=512,
    model_dir="transformer",
    torch_dtype=torch.float32,
    bs=1,
    cfg_mode=True,
):

    transformer_local_dir = get_local_path(local_dir, model_dir)
    onnx_file = "model.onnx"
    onnx_path = os.path.join(transformer_local_dir, onnx_file)
    if os.path.exists(onnx_path):
        return onnx_path
    latent_h, latent_w = (
        img_height // compression_factor,
        img_width // compression_factor,
    )

    if "schnell" in hf_model_path or cfg_mode == False:
        model = FluxModelSchnell(torch_dtype=torch_dtype, model_id=hf_model_path)
        config = model.mmdit.config
        sample_inputs = (
            torch.randn(
                bs,
                (latent_h // 2) * (latent_w // 2),
                config["in_channels"],
                dtype=torch_dtype,
            ),
            torch.randn(bs, max_len, config["joint_attention_dim"], dtype=torch_dtype),
            torch.randn(bs, config["pooled_projection_dim"], dtype=torch_dtype),
            torch.tensor([0.0], dtype=torch.int64),
            torch.randn(100, dtype=torch_dtype),
            torch.empty(bs, dtype=torch_dtype),
        )
    else:
        model = FluxModelCFG(torch_dtype=torch_dtype, model_id=hf_model_path)
        config = model.mmdit.config
        cfg_bs = bs * 2
        sample_inputs = (
            torch.randn(
                cfg_bs,
                (latent_h // 2) * (latent_w // 2),
                config["in_channels"],
                dtype=torch_dtype,
            ),
            torch.randn(
                cfg_bs, max_len, config["joint_attention_dim"], dtype=torch_dtype
            ),
            torch.randn(cfg_bs, config["pooled_projection_dim"], dtype=torch_dtype),
            torch.tensor([0.0], dtype=torch.int64),
            torch.randn(100, dtype=torch_dtype),
            torch.randn(bs, dtype=torch_dtype),
        )

    input_names = ["img", "txt", "vec", "step", "timesteps", "guidance_scale"]

    if not os.path.isfile(onnx_path):
        output_names = ["latent"]

        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample_inputs,
                onnx_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=False,
            )

    assert os.path.isfile(onnx_path)

    return onnx_path


def do_onnx_import(args, model_dir="transformer"):
    if args.save_params_to:
        params_path = args.save_params_to
    else:
        params_path = None
    mlir_path = args.save_mlir_to
    onnx_model_path = os.path.join(args.path, model_dir, "model.onnx")
    process_args = [
        "python",
        "-m",
        "iree.compiler.tools.import_onnx",
        onnx_model_path,
        "-o",
        mlir_path,
        "--externalize-params",
        "--large-model",
        "--num-elements-threshold=32",
    ]
    if params_path:
        process_args.extend(["--save-params-to", params_path])

    subprocess.run(process_args)
    return mlir_path, params_path


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Flux Sampler ONNX export")

    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        choices=["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"],
        help="Model name",
    )
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Precision with which to export the model.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cfg_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether or not to use CFG mode (batch dim -> 2, enables conditioning, flux-dev/pro only)",
    )
    parser.add_argument(
        "--save_mlir_to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_params_to",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    torch_dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    model_dir = "transformer"

    onnx_path = get_flux_sampler_model(
        args.path,
        args.hf_model_id,
        img_height=args.height,
        img_width=args.width,
        compression_factor=8,
        max_len=512,
        model_dir=model_dir,
        torch_dtype=torch_dtypes[args.dtype],
        bs=args.batch_size,
        cfg_mode=args.cfg_mode,
    )
    if args.save_mlir_to or args.save_params_to:
        mlir_path, params_path = do_onnx_import(args, model_dir=model_dir)
