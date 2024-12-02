import os
import torch
import math
from diffusers import FluxTransformer2DModel
from typing import Callable
from iree.turbine.aot import *


def get_local_path(local_dir, model_dir):
    model_local_dir = os.path.join(local_dir, model_dir)
    if not os.path.exists(model_local_dir):
        os.makedirs(model_local_dir)
    return model_local_dir


class FluxModelCFG(torch.nn.Module):
    def __init__(self, torch_dtype):
        super().__init__()
        self.mmdit = FluxTransformer2DModel.from_single_file(
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
        ).to(torch_dtype)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        img_ids,
        txt_ids,
        guidance_vec,
        t_vec,
        t_curr,
        t_prev,
        cfg_scale,
    ):
        pred = self.mmdit(
            hidden_states=hidden_states,
            img_ids=img_ids,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            pooled_projections=pooled_projections,
            timestep=t_vec,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        pred_uncond, pred = torch.chunk(pred, 2, dim=0)
        pred = pred_uncond + cfg_scale * (pred - pred_uncond)
        hidden_states = hidden_states + (t_prev - t_curr) * pred
        return hidden_states


class FluxModelSchnell(torch.nn.Module):
    def __init__(self, torch_dtype):
        super().__init__()
        self.mmdit = FluxTransformer2DModel.from_single_file(
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors"
        ).to(torch_dtype)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        img_ids,
        txt_ids,
        guidance_vec,
        t_vec,
        t_curr,
        t_prev,
        cfg_scale,
    ):
        pred = self.mmdit(
            hidden_states=hidden_states,
            img_ids=img_ids,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            pooled_projections=pooled_projections,
            timestep=t_vec,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        hidden_states = hidden_states + (t_prev - t_curr) * pred
        return hidden_states


@torch.no_grad()
def get_flux_transformer_model(
    hf_model_path,
    img_height=1024,
    img_width=1024,
    compression_factor=8,
    max_len=512,
    torch_dtype=torch.float32,
    bs=1,
):

    latent_h, latent_w = (
        img_height // compression_factor,
        img_width // compression_factor,
    )

    if "schnell" in hf_model_path:
        model = FluxModelSchnell(torch_dtype=torch_dtype)
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
            torch.randn((latent_h // 2) * (latent_w // 2), 3, dtype=torch_dtype),
            torch.randn(max_len, 3, dtype=torch_dtype),
            torch.tensor([1.0] * bs, dtype=torch_dtype),
            torch.tensor([1.0] * bs, dtype=torch_dtype),
            torch.tensor([1.0], dtype=torch_dtype),
            torch.tensor([1.0], dtype=torch_dtype),
            torch.tensor([1.0] * bs, dtype=torch_dtype),
        )
    else:
        model = FluxModelCFG(torch_dtype=torch_dtype)
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
            torch.randn((latent_h // 2) * (latent_w // 2), 3, dtype=torch_dtype),
            torch.randn(max_len, 3, dtype=torch_dtype),
            torch.tensor([1.0] * bs, dtype=torch_dtype),
            torch.tensor([1.0] * cfg_bs, dtype=torch_dtype),
            torch.tensor([1.0], dtype=torch_dtype),
            torch.tensor([1.0], dtype=torch_dtype),
            torch.tensor([1.0] * bs, dtype=torch_dtype),
        )

    input_names = [
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "img_ids",
        "txt_ids",
        "guidance_vec",
        "t_curr",
        "t_prev",
        "cfg_scale",
    ]
    return model, sample_inputs, input_names

    # if not os.path.isfile(onnx_path):
    #     output_names = ["latent"]
    #     dynamic_axes = {
    #         'hidden_states': {0: 'B', 1: 'latent_dim'},
    #         'encoder_hidden_states': {0: 'B',1: 'L'},
    #         'pooled_projections': {0: 'B'},
    #         'timestep': {0: 'B'},
    #         'img_ids': {0: 'latent_dim'},
    #         'txt_ids': {0: 'L'},
    #         'guidance': {0: 'B'},
    #     }

    #     with torch.inference_mode():
    #         torch.onnx.export(
    #             model,
    #             sample_inputs,
    #             onnx_path,
    #             export_params=True,
    #             input_names=input_names,
    #             output_names=output_names)

    # assert os.path.isfile(onnx_path)

    # return onnx_path
