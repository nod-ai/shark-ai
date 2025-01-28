import os
import torch
import math
from diffusers import FluxTransformer2DModel
from typing import Callable
from iree.turbine.aot import *
from sharktank.models.flux.flux import FluxModelV1, FluxParams
from sharktank.types.theta import Theta, Dataset, torch_module_to_theta


def get_local_path(local_dir, model_dir):
    model_local_dir = os.path.join(local_dir, model_dir)
    if not os.path.exists(model_local_dir):
        os.makedirs(model_local_dir)
    return model_local_dir



class FluxDenoiseStepModel(torch.nn.Module):
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.mmdit = FluxModelV1(theta=theta, params=params)
        self.batch_size = batch_size
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_prev = torch.index_select(timesteps, 0, step + 1)
        t_vec = t_curr.repeat(self.batch_size)

        pred = self.mmdit(
            img=img,
            img_ids=self.img_ids,
            txt=txt,
            txt_ids=self.txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        #pred_uncond, pred = torch.chunk(pred, 2, dim=0)
        #pred = pred_uncond + guidance_scale * (pred - pred_uncond)
        img = img + (t_prev - t_curr) * pred
        return img


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
    #transformer_dataset = Dataset.load(transformer_path)
    #transformer_dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/transformer/model.irpa")
    transformer_dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/transformer.irpa")
    model = FluxDenoiseStepModel(theta=transformer_dataset.root_theta, params=FluxParams.from_hugging_face_properties(transformer_dataset.properties))
    #model = FluxModelV1(theta=transformer_dataset.root_theta, params=FluxParams.from_hugging_face_properties(transformer_dataset.properties))
    #dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/transformer.irpa")
    #transformer_params = FluxParams.from_hugging_face_properties(transformer_dataset.properties)
    #model = FluxModelV1(
    #    theta=transformer_dataset.root_theta,
    #    params=transformer_params
    #)
    sample_args, sample_kwargs = model.mmdit.sample_inputs()
    sample_inputs = (
        sample_kwargs["img"],
        #sample_kwargs["img_ids"],
        sample_kwargs["txt"],
        #sample_kwargs["txt_ids"],
        sample_kwargs["y"],
        torch.full((bs,), 1, dtype=torch.int64),
        torch.full((100,), 1, dtype=torch_dtype), # TODO: non-dev timestep sizes
        sample_kwargs["guidance"],
    )
    return model, sample_inputs

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
