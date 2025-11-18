# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team.
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CLIP text model
# It is based on
# https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/clip/modeling_clip.py

from typing import Any, Optional, Tuple, Union
import torch
from torch import nn
import transformers

# TODO: port _prepare_4d_attention_mask and _create_4d_causal_attention_mask to amdsharktank
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _create_4d_causal_attention_mask,
)
from collections import OrderedDict

from amdsharktank.layers import ThetaLayer, LinearLayer, LayerNorm, TokenEmbeddingLayer
from amdsharktank import ops
from amdsharktank.types.theta import Theta
from amdsharktank.types.tensors import AnyTensor
from amdsharktank.layers.configs import ClipTextConfig
from amdsharktank.layers.activations import ACT2FN


class ClipTextEmbeddings(nn.Module):
    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__()
        self.token_embedding = TokenEmbeddingLayer(
            theta=theta("token_embedding"), dtype=None
        )
        self.position_embedding = TokenEmbeddingLayer(
            theta=theta("position_embedding"), dtype=None
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class ClipAttention(ThetaLayer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__(theta)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = LinearLayer(theta("k_proj"))
        self.v_proj = LinearLayer(theta("v_proj"))
        self.q_proj = LinearLayer(theta("q_proj"))
        self.out_proj = LinearLayer(theta("out_proj"))

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = ops.matmul(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class ClipMlp(ThetaLayer):
    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__(theta)
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = LinearLayer(theta("fc1"))
        self.fc2 = LinearLayer(theta("fc2"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class ClipEncoderLayer(ThetaLayer):
    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__(theta)
        self.embed_dim = config.hidden_size
        self.self_attn = ClipAttention(theta=theta("self_attn"), config=config)
        self.layer_norm1 = LayerNorm(
            theta=theta("layer_norm1"), eps=config.layer_norm_eps
        )
        self.mlp = ClipMlp(theta=theta("mlp"), config=config)
        self.layer_norm2 = LayerNorm(
            theta=theta("layer_norm2"), eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ClipEncoder(ThetaLayer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ClipEncoderLayer`].
    """

    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__(theta)
        self.config = config
        self.layers = nn.ModuleList(
            [
                ClipEncoderLayer(theta=theta(f"layers.{i}"), config=config)
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict[str, Any]]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a dict instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return OrderedDict(
            (k, v)
            for k, v in [
                ("last_hidden_state", hidden_states),
                ("hidden_states", encoder_states),
                ("attentions", all_attentions),
            ]
            if v is not None
        )


class ClipTextTransformer(ThetaLayer):
    def __init__(self, theta: Theta, config: ClipTextConfig):
        super().__init__(theta)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = ClipTextEmbeddings(theta=theta("embeddings"), config=config)
        self.encoder = ClipEncoder(theta=theta("encoder"), config=config)
        self.final_layer_norm = LayerNorm(
            theta=theta("final_layer_norm"), eps=config.layer_norm_eps
        )

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict[str, Any]]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = (
            encoder_outputs["last_hidden_state"] if return_dict else encoder_outputs[0]
        )
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR
            # https://github.com/huggingface/transformers/pull/24773
            # Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0]),
                input_ids.argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR
            # https://github.com/huggingface/transformers/pull/24773
            # (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0]),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids == self.eos_token_id).to(torch.int32).argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return OrderedDict(
            (k, v)
            for k, v in [
                ("last_hidden_state", last_hidden_state),
                ("pooler_output", pooled_output),
                (
                    "hidden_states",
                    encoder_outputs["hidden_states"]
                    if "hidden_states" in encoder_outputs
                    else None,
                ),
                (
                    "attentions",
                    encoder_outputs["attentions"]
                    if "attentions" in encoder_outputs
                    else None,
                ),
            ]
            if v is not None
        )


class ClipTextModel(ThetaLayer):
    def __init__(self, *, config: ClipTextConfig, theta: Theta | None = None):
        super().__init__(theta, config=config)
        self.text_model = ClipTextTransformer(
            theta=self.theta("text_model"), config=config
        )

    @classmethod
    def config_type(cls) -> type[ClipTextConfig]:
        return ClipTextConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def sample_inputs(
        self, batch_size: int | None = 1, function: Optional[str] = None
    ) -> tuple[tuple[AnyTensor, ...], OrderedDict[str, AnyTensor]]:
        assert batch_size is not None
        assert function == "forward" or function is None
        sequence_lens = torch.randint(
            low=1, high=self.config.max_position_embeddings + 1, size=(batch_size,)
        )
        input_ids = ops.full(
            size=(batch_size, self.config.max_position_embeddings),
            fill_value=self.config.eos_token_id,
            dtype=torch.long,
        )
        for batch_idx, l in enumerate(sequence_lens):
            input_ids[batch_idx][0:l] = torch.randint(
                low=0, high=self.config.vocab_size - 1, size=(l,), dtype=torch.long
            )
        return tuple(), OrderedDict(
            [
                (
                    "input_ids",
                    input_ids,
                )
            ]
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict[str, Any]]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def load_theta_from_hugging_face(self) -> Theta:
        hugging_face_repo_id = self.config.hugging_face_repo_id
        if hugging_face_repo_id is not None:
            import transformers

            revision = self.config.hugging_face_revision or "main"
            subfolder = self.config.hugging_face_subfolder or ""
            hf_model = transformers.CLIPTextModel.from_pretrained(
                hugging_face_repo_id, revision=revision, subfolder=subfolder
            )
            from .export import hugging_face_clip_text_model_to_theta

            return hugging_face_clip_text_model_to_theta(hf_model)

        raise ValueError(
            "Could not load Theta from Hugging Face. "
            f"Missing config option hugging_face_repo_id"
        )

    def generate_random_theta(self) -> Theta:
        from .export import hugging_face_clip_text_model_to_theta

        hf_config = self.config.to_hugging_face_clip_text_model_config()

        if self.config.rng_seed is not None:
            with torch.random.fork_rng():
                torch.random.manual_seed(self.config.rng_seed)
                model = transformers.CLIPTextModel(hf_config)
        else:
            model = transformers.CLIPTextModel(hf_config)

        return hugging_face_clip_text_model_to_theta(model)
