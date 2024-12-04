### This file contains impls for underlying related models (CLIP, T5, etc)

import torch, math
from torch import nn
from transformers import CLIPTokenizer, T5TokenizerFast
from transformers import T5EncoderModel
from iree.turbine import ops
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sharktank.layers import T5Config
from sharktank.models import t5

CLIP_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3172,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipTextEncoderModule(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        repo,
        precision,
    ):
        super().__init__()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.clip = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device="cpu",
            dtype=self.dtype,
            layer_norm_hidden_state=False,
            return_projected_pooled=True,
            textmodel_json_config=CLIP_CONFIG,
        )
        if precision == "fp16":
            self.clip = self.clip.half()
        clip_weights = hf_hub_download(
            repo_id=repo,
            filename="text_encoder/model.safetensors",
        )
        with safe_open(clip_weights, framework="pt", device="cpu") as f:
            load_into(f, self.clip.transformer, "", "cpu", self.dtype)

    def forward(self, clip_ids):
        vec = self.clip(clip_ids)

        return vec


#################################################################################################
### Core/Utility
#################################################################################################


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    # ops.iree.trace_tensor("attention_q", q[0,0,:5])
    # ops.iree.trace_tensor("attention_k", k[0,0,:5])
    # ops.iree.trace_tensor("attention_v", v[0,0,:5])
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    # ops.iree.trace_tensor("attention_out", out[0,0,:5])
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features, hidden_features, bias=bias, dtype=dtype, device=device
        )
        self.act = act_layer
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias, dtype=dtype, device=device
        )

    def forward(self, x):
        x = self.fc1(x)
        # ops.iree.trace_tensor("mlpfx", x[0,0,:5])
        x = self.act(x)
        # ops.iree.trace_tensor("mlpact", x[0,0,:5])
        x = self.fc2(x)
        # ops.iree.trace_tensor("mlpanotherfc", x[0,0,:5])
        return x


def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


#################################################################################################
### CLIP
#################################################################################################


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}


class CLIPLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        # self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        self.mlp = Mlp(
            embed_dim,
            intermediate_size,
            embed_dim,
            act_layer=ACTIVATIONS[intermediate_activation],
            dtype=dtype,
            device=device,
        )

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                CLIPLayer(
                    embed_dim,
                    heads,
                    intermediate_size,
                    intermediate_activation,
                    dtype,
                    device,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(
        self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(
        self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True
    ):
        x = self.embeddings(input_tokens)
        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        x, i = self.encoder(
            x, mask=causal_mask, intermediate_output=intermediate_output
        )
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class SDTokenizer:
    def __init__(
        self,
        max_length=77,
        pad_with_end=True,
        tokenizer=None,
        has_start_token=True,
        pad_to_max_length=True,
        min_length=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str):
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here. The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(" ")
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend(
                [
                    (t, 1)
                    for t in self.tokenizer(word)["input_ids"][self.tokens_start : -1]
                ]
            )
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text: str | list[str]):
        out = {}
        if isinstance(text, list):
            text = text[0]
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        for k, v in out.items():
            out[k] = torch.tensor(v, dtype=torch.int64, device="cpu")
        return out


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        # tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        tokens = token_weight_pairs[:, :, 0]
        out, pooled = self(tokens)
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        device="cpu",
        max_length=77,
        layer="last",
        layer_idx=None,
        textmodel_json_config=None,
        dtype=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        return_projected_pooled=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def encode_token_weights(self, token_weight_pairs):
        pass

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, token_weight_pairs):
        # tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        tokens = token_weight_pairs[:, :, 0]
        # backup_embeds = self.transformer.get_input_embeddings()
        # device = backup_embeds.weight.device
        # tokens = torch.LongTensor(tokens).to(device)
        outputs = self.transformer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        # self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        out, pooled = z.float(), pooled_output
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(
        self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )
