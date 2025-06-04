from sharktank.models.llama.testing import make_random_llama_theta 
from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.types import Dataset
from sharktank.examples.export_paged_llm_v1 import PagedLlmModelV1

import argparse
import torch
from dataclasses import asdict

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="./toy_llama4.irpa")

def generate(seed):
    torch.manual_seed(seed=12345)
    dtype = torch.float16

    attention_head_count_kv = 4
    attention_head_count = attention_head_count_kv * 5
    vocabulary_size = 19

    # When comparing with Hugging Face, its Flex attention requires a power 2.
    rope_dimension_count = 4 * 2

    attn_head_dim = rope_dimension_count
    block_seq_stride = 13
    block_count = 4
    rope_layers = [i for i in range(block_count) if int((i + 1) % 4 != 0)]
    expert_feed_forward_length = 29
    config=LlamaModelConfig(
        hp=LlamaHParams(
            context_length=block_seq_stride * 11,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=4,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=500000.0,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=0.01,
            attention_head_count_kv=attention_head_count_kv,
            expert_feed_forward_length=expert_feed_forward_length,
            expert_count=3,
            expert_used_count=2,
            expert_shared_count=1,
            expert_shared_feed_forward_length=expert_feed_forward_length,
            interleave_moe_layer_step=2,
            model_arch="llama4",
            vocab_size=vocabulary_size,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        #activation_fn="silu",
        attention_dtype=dtype,
        dtype=dtype,
        use_qk_norm=True,
        #rope_type="llama4",
        rope_layers=[1,3],
        attention_chunk_size=37,
        attn_temperature_tuning=True,
        floor_scale=31,
        attn_scale=0.2,
        #ffn_add_residual=True,
    )

    theta=make_random_llama_theta(config=config, vocab_size=vocabulary_size)
    print('here is theta: ',theta)
    model = PagedLlmModelV1(theta, config=config)
    print(model.attn_blocks)

    return theta, config

def main():
    args = parser.parse_args()
    theta, config = generate(args.seed)
    for tensor in theta.flatten().values():
        print(f"  {tensor.name}: {tensor.shape}, {tensor.dtype}")

    config_dict = {
        "hparams": asdict(config.hp),
    }

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)

if __name__=="__main__":
    main()
