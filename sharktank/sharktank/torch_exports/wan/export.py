
import argparse
import logging
import time
from typing import Dict, Any
from pathlib import Path
from dataclasses import fields
import functools
import os

import torch
import torch.nn.functional as F
from iree.turbine.aot import *
from iree.turbine import aot, ops
import numpy as np
from sharktank.types.theta import torch_module_to_theta, Dataset
from sharktank.transforms.dataset import set_float_dtype

import time

from clip import clip_xlm_roberta_vit_h_14

# Global variables for models
torch.random.manual_seed(0)
ARTIFACTS_DIR = "."
BATCH_SIZE = 1

height = 512
width = 512

config = None
text_clip_model = None
score_model = None
rank = 0


def load_model_components(args):
    """Initialize models with random weights."""
    device = "cpu"
    #torch.cuda.set_device(rank)
    pass

def transform_normalize(x):
    mean = torch.as_tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.bfloat16).view(-1, 1, 1)
    std = torch.as_tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.bfloat16).view(-1, 1, 1)
    # ops.iree.trace_tensor("mean", mean)
    return x.sub_(mean).div_(std)
    
class ExportSafeClipModel(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        # init model
        self.model = mod
        self.size = (self.model.image_size, self.model.image_size)
        
        # Don't load real weights.
        # logging.info(f'loading {checkpoint_path}')
        # self.model.load_state_dict(
        #     torch.load(checkpoint_path, map_location='cpu'))

    def forward(self, video_1, video_2):
        videos = [
            F.interpolate(
                video_1.transpose(0, 1).type(torch.float16),
                size=self.size,
                mode='bicubic',
                align_corners=False),
            F.interpolate(
                video_2.transpose(0, 1).type(torch.float16),
                size=self.size,
                mode='bicubic',
                align_corners=False),
        ]
        videos = torch.cat(videos)
        videos = transform_normalize(videos.mul_(0.5).add_(0.5)).to(torch.bfloat16)
        # forward
        out = self.model.visual(videos, use_31_block=True)
        return out
    
def get_clip_visual_model_and_inputs():
    global height, width
    inner = clip_xlm_roberta_vit_h_14(
                pretrained=False,
                return_transforms=False,
                return_tokenizer=False,
                dtype=torch.bfloat16,
            ).eval().requires_grad_(False)
    mod = ExportSafeClipModel(inner).eval().requires_grad_(False)
    mod.model.log_scale = torch.nn.Parameter(mod.model.log_scale.to(torch.float32)).requires_grad_(False)
    inputs = {
        "forward": {
            "video_1": torch.rand(3, 1, height, width, dtype=torch.float16),
            "video_2": torch.rand(3, 1, height, width, dtype=torch.float16),
        }
    }
    np.save("clip_input1.npy", np.asarray(inputs["forward"]["video_1"]).astype("float16"))
    np.save("clip_input2.npy", np.asarray(inputs["forward"]["video_2"]).astype("float16"))
    return mod, inputs

def filter_properties_for_config(
    properties: Dict[str, Any], config_class: Any
) -> Dict[str, Any]:
    """Filter properties to only include fields valid for the given config class.

    Args:
        properties: Properties dictionary
        config_class: The dataclass to filter properties for

    Returns:
        Filtered properties dictionary with only valid fields for the config class
    """
    # Start with hparams if available
    if "hparams" in properties:
        props = properties["hparams"]
    else:
        props = properties

    # Get set of valid field names for the config class
    valid_fields = {f.name for f in fields(config_class)}

    # Filter to only include valid fields
    filtered_props = {k: v for k, v in props.items() if k in valid_fields}

    return filtered_props

def get_t5_text_model_and_inputs():
    from sharktank.models.t5.export import import_encoder_dataset_from_hugging_face
    from sharktank.models.t5 import T5Config, T5Encoder

    model_path = "google/umt5-xxl"
    dtype_str = "bf16"
    output_path = Path(ARTIFACTS_DIR)
    t5_path = Path(model_path)
    t5_tokenizer_path = Path(model_path)
    t5_output_path = output_path / f"wan2_1_umt5xxl_{dtype_str}.irpa"
    t5_dataset = import_encoder_dataset_from_hugging_face(
        str(t5_path), tokenizer_path_or_repo_id=str(t5_tokenizer_path)
    )
    t5_dataset.properties = filter_properties_for_config(
        t5_dataset.properties, T5Config
    )
    t5_dataset.save(str(t5_output_path))

    class HFEmbedder(torch.nn.Module):
        def __init__(self, dataset, max_length: int):
            super().__init__()
            self.max_length = max_length
            self.output_key = "last_hidden_state"

            t5_config = T5Config.from_properties(dataset.properties)
            self.hf_module = T5Encoder(theta=dataset.root_theta, config=t5_config)

            self.hf_module = self.hf_module.eval().requires_grad_(False)

        def forward(self, input_ids) -> torch.Tensor:

            outputs = self.hf_module(
                input_ids=input_ids,
                attention_mask=None,
                output_hidden_states=False,
            )
            return outputs[self.output_key]
    
    t5_mod = HFEmbedder(
        t5_dataset,
        512,
    )
    t5_sample_inputs = {
        "forward": {
            "input_ids": torch.ones([BATCH_SIZE, 512], dtype=torch.int64),
        }
    }
    t5_output = t5_mod.forward(t5_sample_inputs["forward"]["input_ids"])
    np.save("umt5xxl_input.npy", np.asarray(t5_sample_inputs["forward"]["input_ids"]))

    np.save("umt5xxl_output.npy", np.asarray(t5_output.to(torch.float16).detach()))
    return t5_mod, t5_sample_inputs

# The c.ai benchmark script does not run through the wav2vec model. Skip for now.
# def get_wav2vec_model_and_inputs():
#     global audio_model
#     audio_mod = audio_model
#     audio_inputs = [torch.tensor()]
#     return audio_mod, audio_inputs

class WanVaeWrapped(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.model = mod

    def encode(self, x):
        x = x.to(torch.bfloat16)
        return self.model.encode(x).latent_dist.mode()
    
    def decode(self, z):
        z = z.to(torch.bfloat16)
        return self.model.decode(z, return_dict=False)


def get_vae_model_and_inputs():
    # from shark_wanvae import SanitizedWanVAE
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0
    )
    # mod = AutoencoderKLWan.from_pretrained(
    #     "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    #     subfolder="vae",
    #     torch_dtype=torch.bfloat16  
    # ).requires_grad_(False).eval()
    # model = WanVaeWrapped(mod).requires_grad_(False).eval()
    # model = SanitizedWanVAE(**cfg).bfloat16().to("cpu").requires_grad_(False).eval()
    from orig_vae import WanVAE_
    #scale = torch.tensor(scale_py, dtype=torch.float16)
    model = WanVAE_(**cfg).bfloat16().to("cuda").requires_grad_(False).eval()
    inputs = {
        "encode": {
            "x": torch.rand(1, 3, 1, height, width, dtype=torch.float16),
        },
        "decode": {
            "z": torch.rand(1, 16, 1, height, width, dtype=torch.float16),
        }
    }
    np.save("vae_encode_input.npy", np.asarray(inputs["encode"]["x"]).astype("float16"))
    np.save("vae_decode_input.npy", np.asarray(inputs["decode"]["z"]).astype("float16"))
    model.to("cuda:0")
    enc_start = time.time()
    vae_enc_output = model.encode(inputs["encode"]["x"].to("cuda")).clone().detach()
    print("ENCODE LATENCY: ", str(time.time() - enc_start), " seconds")
    dec_start = time.time()
    vae_dec_output = model.decode(inputs["decode"]["z"].to("cuda")).clone().detach()
    print("DECODE LATENCY: ", str(time.time() - dec_start), " seconds")

    
    np.save("vae_encode_output.npy", np.asarray(vae_enc_output.to(torch.float16)))
    np.save("vae_decode_output.npy", np.asarray(vae_dec_output.to(torch.float16)))

    return model, inputs

def export_model_components(args):
    # clip_artifacts = ["wan2_1_clip_512x512.mlir", "wan2_1_clip_bf16.irpa"]
    # if not artifacts_exist(clip_artifacts) or "clip" in args.force_export:
    #     print("Exporting CLIP model...")
    #     clip_mod, clip_inputs = get_clip_visual_model_and_inputs()
    #     export_model_mlir(
    #         clip_mod, 
    #         clip_artifacts[0], 
    #         clip_inputs, 
    #         decomp_attn=True, 
    #         weights_filename=clip_artifacts[1]
    #     )
    t5_artifacts = ["wan2_1_umt5xxl.mlir", "wan2_1_umt5xxl_bf16.irpa"]
    if not artifacts_exist(t5_artifacts) or "t5" in args.force_export:
        print("Exporting umt5-xxl model...")
        t5_mod, t5_inputs = get_t5_text_model_and_inputs()
        export_model_mlir(
            t5_mod, 
            t5_artifacts[0], 
            t5_inputs, 
            weights_filename=t5_artifacts[1]
        )
    vae_artifacts = ["wan2_1_vae_512x512.mlir", "wan2_1_vae_bf16.irpa"]
    if not artifacts_exist(vae_artifacts) or "vae" in args.force_export:
        print("Exporting VAE model...")
        vae_mod, vae_inputs = get_vae_model_and_inputs()
        export_model_mlir(vae_mod, vae_artifacts[0], vae_inputs, decomp_attn=True, weights_filename=vae_artifacts[1])

def artifacts_exist(artifacts: list) -> bool:
    for artifact in artifacts:
        if not os.path.exists(artifact):
            return False
    return True

def save_dataset(path, model):
    theta = torch_module_to_theta(model)
    theta.rename_tensors_to_paths()
    theta.transform(
        functools.partial(set_float_dtype, dtype=torch.bfloat16)
    )
    ds = Dataset(root_theta=theta, properties={})
    ds.save(path)

def export_model_mlir(
    model,
    output_path,
    function_inputs_map,
    decomp_attn = False,
    weights_filename = "model.irpa"
):
    """Export a model with no dynamic dimensions.

    For the set of provided function name batch sizes pair, the resulting MLIR will
    have function names with the below format.
    ```
    <function_name>_bs<batch_size>
    ```

    If `batch_sizes` is given then it defaults to a single function with named
    "forward".

    The model is required to implement method `sample_inputs`.
    """
    decomp_list = [
        torch.ops.aten.logspace, 
        torch.ops.aten.upsample_bicubic2d.vec,
        torch.ops.aten._upsample_nearest_exact2d.vec,
        torch.ops.aten.as_strided,
        torch.ops.aten.as_strided_copy.default
    ]
    if decomp_attn:
        decomp_list.extend([
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ])
    with aot.decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        save_dataset(weights_filename, model)
        aot.externalize_module_parameters(model)

        fxb = aot.FxProgramsBuilder(model)

        for function, input_kwargs in function_inputs_map.items():
            @fxb.export_program(
                name=f"{function or 'forward_bs1'}",
                args=(),
                kwargs=input_kwargs,
                strict=False,
            )
            def _(model, **kwargs):
                return getattr(model, function, model.forward)(**kwargs)

        output = aot.export(fxb)
    output.save_mlir(output_path)
    print("Saved MLIR to: ", str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Benchmark WANI2V Video Generation")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--warmup", action="store_true", help="Warmup model")
    parser.add_argument("--prompt", type=str, default="A person talking", help="Text prompt")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_export", type=str, default="", help="model to force new export for. Comma-separated t5, clip, vae",)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load models
    load_model_components(args)
    
    # Export models
    export_model_components(args)

    # Prepare fake inputs
    fake_image_tensor = torch.rand(3, args.height, args.width, dtype=torch.bfloat16, device=rank) * 2 - 1

    # if args.warmup:
    #     run(fake_image_tensor, args)

    # # Run benchmark
    # torch.cuda.synchronize()
    start_time = time.time()
    
    # run(fake_image_tensor, args)
    
    # torch.cuda.synchronize()
    elapsed = time.time() - start_time
    logging.info(f"Inference time: {elapsed:.3f} seconds")

if __name__ == "__main__":
    main()
