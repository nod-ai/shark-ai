import torch
from torch import Tensor, nn

from sharktank.types.theta import Theta, Dataset, torch_module_to_theta
from transformers import CLIPTextModel
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.models.t5 import T5Encoder, T5Config

# Copied from https://github.com/black-forest-labs/flux
class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
            #theta = torch_module_to_theta(self.hf_module)
            config = ClipTextConfig.from_hugging_face_clip_text_model_config(self.hf_module.config)
            config.dtype = torch.float32
            dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/clip.irpa")
            self.hf_module = ClipTextModel(theta=dataset.root_theta, config=config)
        else:
            t5_dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/t5.irpa")
            t5_config = T5Config.from_gguf_properties(
                t5_dataset.properties,
                feed_forward_proj="gated-gelu",
            )
            self.hf_module = T5Encoder(theta=t5_dataset.root_theta, config=t5_config)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, input_ids) -> Tensor:
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
