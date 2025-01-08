from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

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
        else:
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, input_ids) -> Tensor:
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
