from parameterized import parameterized
import transformers.models
from sharktank.utils.testing import TempDirTestBase
from sharktank.models.llama4.testing import (
    make_toy_model_config,
    config_to_hugging_face_text_config,
    theta_to_hugging_face_state_dict,
)
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llm import PagedLlmModelV1
import transformers

import torch


def convert_hf_2D_input_mask_to_4D_attention_mask(
    mask: torch.Tensor, model: PagedLlmModelV1
) -> torch.Tensor:
    inverted_mask = mask == 0
    return model.attention_mask(inverted_mask)


class Llama4Test(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    @parameterized.expand(
        [
            (torch.float32, 1e-5),
        ]
    )
    def testCompareToyEagerVsHuggingFace(self, dtype: torch.dtype, atol: float):
        torch.set_printoptions(
            linewidth=120, threshold=1000, edgeitems=4, precision=2, sci_mode=True
        )

        config = make_toy_model_config(dtype=dtype)
        theta = make_random_llama_theta(config, dtype=dtype)
        hf_config = config_to_hugging_face_text_config(config)

        model = PagedLlmModelV1(theta=theta, config=config)
        hf_model = transformers.models.llama4.Llama4ForCausalLM(hf_config)

        orig_state_dict = hf_model.state_dict()
        hf_state_dict = theta_to_hugging_face_state_dict(theta, config)
        hf_model.load_state_dict(hf_state_dict)

        batch_size = 41
        batch_seq_len = config.hp.context_length
        input_ids = torch.randint(
            low=0,
            high=config.vocabulary_size,
            size=[batch_size, batch_seq_len],
            dtype=torch.long,
        )
        # inputs_embeds = torch.rand(size=[batch_size, batch_seq_len, config.hp.embedding_length], dtype=dtype)
        # We need to create the cache ourselves as HF would create it always in bf16.
        hf_past_key_values = transformers.cache_utils.HybridChunkedCache(
            hf_config,
            max_batch_size=input_ids.shape[0],
            max_cache_len=input_ids.shape[1],
            dtype=dtype,
        )

        hf_2d_attention_mask = torch.randint_like(input_ids, low=0, high=2)
        attention_mask = convert_hf_2D_input_mask_to_4D_attention_mask(
            mask=hf_2d_attention_mask, model=model
        )

        from sharktank.utils.patching import SaveModuleResultTensorsPatch

        hf_intermediates_saver = SaveModuleResultTensorsPatch(with_before_forward=True)
        hf_intermediates_saver.patch_child_modules(hf_model)

        @torch.compiler.disable(recursive=True)
        def run_hf_model():
            return hf_model(
                input_ids=input_ids,
                attention_mask=hf_2d_attention_mask,
                past_key_values=hf_past_key_values,
            )

        hf_output = run_hf_model()

        page_count = (len(input_ids[0]) // config.block_seq_stride) * batch_size
        kv_cache_state = model.cache.allocate(page_count)
        seq_block_ids = torch.arange(
            start=0, end=input_ids.numel() // config.block_seq_stride, dtype=torch.long
        ).view(batch_size, batch_seq_len // config.block_seq_stride)

        intermediates_saver = SaveModuleResultTensorsPatch(with_before_forward=True)
        intermediates_saver.patch_child_modules(model)

        model.prefill(
            tokens=input_ids,
            attention_mask=attention_mask,
            cache_state=kv_cache_state,
            seq_block_ids=seq_block_ids,
        )

        hf_intermediates_saver.save_file(
            "hf_trace.safetensors", skip_unsupported_dtypes=True
        )
        intermediates_saver.save_file("trace.safetensors", skip_unsupported_dtypes=True)

    def test_moe(self):
        from sharktank.layers.testing import make_random_moe_block_theta
        from sharktank.layers import MoeBlock

        dtype = torch.float32
        feature_dim = 7
        expert_hidden_dim = 3
        num_experts = 5
        expert_used_count = 2
        num_shared_experts = 11
        shared_expert_hidden_dim = 13
        batch_size = 17
        sequence_length = 19

        theta = make_random_moe_block_theta(
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            shared_expert_hidden_dim=shared_expert_hidden_dim,
            with_layer_output_norm=True,
            dtype=dtype,
        )

        moe_block = MoeBlock(
            theta=theta,
            expert_used_count=expert_used_count,
            rms_epsilon=0.01,
            moe_activation=torch.nn.functional.silu,
            score_experts=torch.nn.functional.sigmoid,
            normalize_experts=False,
            add_residual=False,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        moe_block(input)
