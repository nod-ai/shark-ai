from sharktank.utils.iree import (
    with_iree_device_context,
    load_iree_module,
    prepare_iree_module_function_args,
    flatten_for_iree_signature,
)

from sharktank.layers.configs import LlamaModelConfig
from sharktank.models.llm import PagedLlmModelV1
from sharktank.models.llama4.toy_llama4 import generate
from sharktank.types import *
from sharktank.utils.evaluate import *
from sharktank.utils.iree import (
    get_iree_devices,
    make_hal_buffer_view_trace_default_callback,
    TorchLikeIreeModule,
)
from sharktank.utils.export_artifacts import ExportArtifacts
from sharktank.examples import export_paged_llm_v1
from sharktank.layers.mixture_of_experts_block import MoeBlock
from sharktank.layers.norm import L2Norm
from sharktank.ops import topk, zeros_like, reshard_like

import torch
import logging
from os import PathLike
from dataclasses import asdict
import iree.runtime
from collections import OrderedDict
import argparse
from pathlib import Path
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

iree_compile_flags = []  # TODO; fill this if we need any flag


def _sample_inputs(self, batch_size=1, dtype=torch.int64):
    return llama4_toy_sample_inputs(
        self,
        config=self.config,
        batch_size=batch_size,
        dtype=dtype,
    )


def deterministic_attn_mask(inverted_mask):
    L = inverted_mask.shape[-1]
    causal = torch.tril(
        torch.ones(L, L, dtype=inverted_mask.dtype, device=inverted_mask.device)
    )
    return causal.unsqueeze(0)  # (1, L, L)


def deterministic_mask(inverted_mask):
    seq_len = inverted_mask.shape[-1]
    causal = torch.tril(
        torch.ones(
            seq_len, seq_len, dtype=inverted_mask.dtype, device=inverted_mask.device
        )
    )
    return causal.unsqueeze(0)


def deterministic_input_mask(seq_lens, sl, *, device="cpu"):
    return torch.zeros(seq_lens.size(0), sl, dtype=torch.bool, device=device)


def deterministic_page_table(page_count, config, dtype=torch.float16):

    sub_page_dims = (
        config.hp.attention_head_count_kv,  # 4
        config.hp.expert_used_count,  # 2
        config.hp.block_count,  # 4
        config.block_seq_stride,  # 13
        config.hp.attn_head_dim,  # 8
    )
    flat_dim = math.prod(sub_page_dims)
    table = torch.arange(page_count * flat_dim, dtype=dtype).view(page_count, flat_dim)

    return [table]


def llama4_toy_sample_inputs(
    model: PagedLlmModelV1,
    config: LlamaModelConfig,
    batch_size: int = 1,
    dtype=torch.int64,
) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:

    hp = config.hp
    context_len = hp.context_length
    base = torch.arange(context_len) % hp.vocab_size
    tokens = base.repeat(batch_size, 1).long()

    page_count = (len(tokens[0]) // config.block_seq_stride) * batch_size
    kv_cache_state = deterministic_page_table(
        page_count, config
    )  # model.cache.allocate(page_count)

    seq_block_ids = torch.arange(
        start=0, end=tokens.numel() // config.block_seq_stride, dtype=dtype
    ).view(batch_size, context_len // config.block_seq_stride)

    hf_2d_attention_mask = torch.ones_like(tokens, dtype=dtype)
    inverted_mask = hf_2d_attention_mask == 0
    attention_mask = deterministic_mask(
        inverted_mask
    )  # model.attention_mask(inverted_mask)
    seq_lens = torch.full((batch_size,), context_len, dtype=dtype)

    args = tuple()
    kwargs = OrderedDict(
        (
            ("tokens", tokens),
            ("seq_lens", seq_lens),
            ("attention_mask", [attention_mask]),
            ("seq_block_ids", [seq_block_ids]),
            ("cache_state", kv_cache_state),
        )
    )
    return args, kwargs


def patch_forward_L2Norm(self, x):
    dtype = torch.float32
    x_f = x.to(dtype)

    sqr = x_f.pow(2)

    dims = self.dim if isinstance(self.dim, tuple) else (self.dim,)
    dims = tuple(d if d >= 0 else x.dim() + d for d in dims)

    permute_dims = [i for i in range(x_f.dim()) if i not in dims] + list(dims)
    x_perm = x_f.permute(permute_dims)
    sqr_perm = sqr.permute(permute_dims)

    kept_shape = x_perm.shape[: -len(dims)]
    x_flat = x_perm.reshape(*kept_shape, -1)
    sqr_flat = sqr_perm.reshape(*kept_shape, -1)

    var = torch.zeros_like(sqr_flat[..., :1])
    for i in range(sqr_flat.shape[-1]):
        var += sqr_flat[..., i : i + 1]
    var = var / sqr_flat.shape[-1]

    out_flat = x_flat * torch.rsqrt(var + self.epsilon)

    out_perm = out_flat.reshape(*x_perm.shape)
    invert_perm = [permute_dims.index(i) for i in range(x_f.dim())]
    out = out_perm.permute(invert_perm)
    return out.to(dtype=x.dtype)


def patch_forward_moe(
    self,
    h: torch.Tensor | ShardedTensor,
):
    batch_size, sequence_length, feature_dim = h.shape
    ffn_input = h.view(-1, feature_dim)

    # For each token, the router calculates the router weights for all experts
    # shape: (batch_size * sequence_length, expert_count)
    router_logits = self.ffn_gate_inp(ffn_input)
    router_weights = self.score_experts(router_logits.to(torch.float))

    router_weights = reshard_like(router_weights, like=ffn_input)

    def pick_first_expert(scores: torch.Tensor, k: int = 1):

        shape = scores.shape[:-1] + (k,)
        top_idx = torch.zeros(shape, dtype=torch.long, device=scores.device)
        top_gate = torch.ones(shape, dtype=scores.dtype, device=scores.device)
        return top_gate, top_idx

    # Select top k experts from router weights
    if self.n_expert_groups is not None and self.n_limited_groups is not None:
        scores_for_choice = router_weights.view(-1, self.expert_count)

        group_scores = (
            router_weights.view(
                -1, self.n_expert_groups, self.expert_count // self.n_expert_groups
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = topk(group_scores, k=self.n_limited_groups, dim=-1)[1]
        group_mask = zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_expert_groups, self.expert_count // self.n_expert_groups)
            .reshape(-1, self.expert_count)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        # shape: (batch_size * sequence_length, expert_used_count)
        expert_gate, top_k_experts = topk(
            scores_for_choice, k=self.expert_used_count, dim=-1
        )
    else:
        # shape: (batch_size * sequence_length, expert_used_count)
        """expert_gate, top_k_experts = topk(
            router_weights, self.expert_used_count, dim=-1
        )"""
        expert_gate, top_k_experts = pick_first_expert(
            router_weights, k=self.expert_used_count
        )

    if self.normalize_experts:
        expert_gate /= expert_gate.sum(dim=-1, keepdim=True)

    expert_gate = expert_gate.to(ffn_input.dtype)

    if self.route_scale is not None:
        expert_gate = expert_gate * self.route_scale

    # shape: (batch_size * sequence_length, feature_dim)
    moe_output = self.routed_experts(ffn_input, top_k_experts, expert_gate)

    if self.expert_shared_count is not None:
        moe_output = moe_output + self.shared_experts(ffn_input)

    moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)

    moe_output = self.layer_output_norm(moe_output)

    return moe_output


def export_llama4_model_mlir(
    model_or_parameters_path: PagedLlmModelV1 | PathLike,
    /,
    output_path: PathLike,
    batch_size: int,
    config: LlamaModelConfig,  # TODO; it can be read from HF when comparing the actual model
):
    mlir_path = output_path / "model.mlir"
    export_config_path = output_path / "model_export_config.json"
    irpa_path = output_path / "parameters.irpa"
    export_artifacts = ExportArtifacts.from_config(
        config,
        irpa_path=irpa_path,
        batch_size=batch_size,
        iree_hip_target="gfx942",
        iree_hal_target_device="hip",
        # iree_hal_local_target_device_backends=self.iree_hal_local_target_device_backends,
        use_attention_mask=True,
    )

    cli_args = [
        "export_paged_llm_v1",
        "--irpa-file",
        str(irpa_path),
        "--output-mlir",
        str(mlir_path),
        "--output-config",
        str(export_config_path),
        "--bs-prefill",
        str(batch_size),
        "--bs-decode",
        str(batch_size),
        "--block-seq-stride",
        str(config.block_seq_stride),
        "--attention-dtype",
        "float16",
        "--activation-dtype",
        "float16",
        "--tensor-parallelism-size",
        "1",
        "--pipeline-parallelism-size",
        "1",
        "--attention-kernel",
        "torch",
        "--skip-decode",
        "--use-attention-mask",
    ]

    sys_argv_backup = sys.argv
    sys.argv = cli_args
    export_paged_llm_v1.main()

    sys.argv = sys_argv_backup

    iree_module_path = output_path / "model.vmfb"
    export_artifacts.compile_to_vmfb(
        output_mlir=mlir_path,
        output_vmfb=iree_module_path,
        args=[],
    )


def export_llama4_iree_parameters(
    model: PagedLlmModelV1,
    parameters_output_path: PathLike,
    config: LlamaModelConfig,  # TODO; it can be read from HF when comparing the actual model
):
    config_dict = {
        "hparams": asdict(config.hp),
    }

    dataset = Dataset(config_dict, root_theta=model.theta)
    dataset.save(parameters_output_path)


def export_llama4(
    model: PagedLlmModelV1,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    output_path: PathLike,
    batch_size: int,
    config: LlamaModelConfig,  # TODO; it can be read from HF when comparing the actual model
):
    export_llama4_iree_parameters(model, parameters_output_path, config=config)
    export_llama4_model_mlir(
        model, output_path=output_path, config=config, batch_size=batch_size
    )


def runCompareIreeAgainstTorchEager(args, atol):
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)

    work_dir = Path(args.work_dir)

    MoeBlock.forward = patch_forward_moe
    L2Norm.forward = patch_forward_L2Norm
    # add sample_inputs
    PagedLlmModelV1.sample_inputs = _sample_inputs
    PagedLlmModelV1.input_mask = staticmethod(deterministic_input_mask)
    PagedLlmModelV1.attention_mask = staticmethod(deterministic_attn_mask)

    target_theta, target_config = generate(seed)
    target_torch_model = PagedLlmModelV1(
        theta=target_theta,
        config=target_config,
    )

    _, kw = target_torch_model.sample_inputs(args.batch_size)
    logger.info("running eager prefill for llama4...")
    expected_outputs = target_torch_model.prefill(
        kw["tokens"],
        attention_mask=kw["attention_mask"],
        seq_block_ids=kw["seq_block_ids"],
        cache_state=kw["cache_state"],
    )

    # Iree model
    mlir_path = work_dir / "model.mlir"
    parameters_path = work_dir / "parameters.irpa"

    logger.info("Exporting llama4 to MLIR...")
    export_llama4(
        target_torch_model,
        output_path=work_dir,
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_path,
        batch_size=args.batch_size,
        config=target_config,
    )

    iree_devices = get_iree_devices(
        device="hip://5",
        device_count=1,
    )
    iree_module_path = work_dir / "model.vmfb"

    def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
        cpu_device = get_iree_devices(driver="local-task", device_count=1)
        iree_buffere_view_trace_callback = make_hal_buffer_view_trace_default_callback(
            cpu_device[0]
        )
        debug_sink = iree.runtime.HalModuleDebugSink(iree_buffere_view_trace_callback)

        iree_module, vm_context, vm_instance = load_iree_module(
            module_path=str(iree_module_path),
            devices=iree_devices,
            parameters_path=str(parameters_path),
            debug_sink=debug_sink,
        )
        iree_kw = {
            "tokens": kw["tokens"],
            "seq_lens": kw["seq_lens"],
            "seq_block_ids": kw["seq_block_ids"][0],
            "cache_state": kw["cache_state"],
        }

        flat_args = flatten_for_iree_signature(iree_kw)

        torch_like_iree_module = TorchLikeIreeModule(
            module=iree_module, devices=iree_devices, vm_context=vm_context
        )
        """iree_result = run_iree_module_function(
            module        = iree_module,
            vm_context    = vm_context,
            args          = iree_args,
            device        = iree_devices[0],
            function_name = "prefill_bs1",
            )"""

        iree_kw_tmp = (
            kw["tokens"],
            kw["seq_lens"],
            kw["seq_block_ids"][0],
            kw["cache_state"],
        )

        iree_result = getattr(torch_like_iree_module, f"prefill_bs{args.batch_size}")(
            *iree_kw_tmp
        )

        # Make sure we don't leak IREE-backed tensors outside of this function.
        iree_result = [t.clone() for t in iree_result]
        iree_logits = iree_result[0]
        return iree_logits

    actual_outputs = with_iree_device_context(run_iree_module, iree_devices)

    logger.info("Comparing outputs...")

    abs_diff = (actual_outputs[0] - expected_outputs[0]).abs()

    print("max error: ", abs_diff.max())
    logger.info(f"Actual vs expected abs diff {(abs_diff)}")
    logger.info(f"max abs error between actual vs expected {(abs_diff.max())}")
    torch.testing.assert_close(
        actual_outputs,
        expected_outputs,
        atol=atol,
        rtol=0,
        msg=f"Actual vs expected results diff > {atol}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    runCompareIreeAgainstTorchEager(args, atol=1e-1)
