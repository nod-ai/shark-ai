import subprocess
import torch
import numpy as np

import torch.nn.functional as F
from iree.turbine.aot import *
from sharktank.layers.base import ThetaLayer
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Theta
from sharktank.utils.export_artifacts import (
    IreeBenchmarkException,
    IreeCompileException,
)


class ScatterLayer(ThetaLayer):
    def generate_random_theta(self):
        return

    def forward(
        self,
        h: torch.Tensor,
        top_experts_index: torch.Tensor,
        expert_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (batch_size * sequence_length, input_feature_dim)
        top_experts_index: (batch_size * sequence_length, num_top_experts)
        expert_gate: (batch_size * sequence_length, num_top_experts)
        """
        num_experts = 4
        num_tokens, input_feature_dim = h.shape

        # (num_experts, num_tokens)
        router_scores = torch.zeros([num_tokens, num_experts], dtype=h.dtype)
        router_scores = router_scores.scatter_(
            1, top_experts_index, expert_gate
        ).transpose(0, 1)

        # NOTE: This one also doesn't work, but likely a different reason (since it doesn't emit a scatter op)
        # one_hot_expert_indices = F.one_hot(
        #     top_experts_index, num_classes=num_experts
        # ).to(dtype=h.dtype, device=h.device)
        # weighted_scores = one_hot_expert_indices * expert_gate.unsqueeze(-1)
        # router_scores = weighted_scores.sum(dim=1).transpose(0, 1)

        # (num_experts * num_tokens, input_feature_dim)
        router_indices = torch.arange(num_tokens).view(1, -1).expand(num_experts, -1)
        router_indices = router_indices.reshape(-1, 1).expand(-1, input_feature_dim)

        routed_in = router_indices.to(dtype=h.dtype)
        routed_out = routed_in * (router_scores > 0).reshape(-1, 1)
        # (num_tokens, input_feature_dim)
        return routed_out[: h.shape[0], ...]


import os

cwd = os.path.realpath(os.curdir)
cwd = "/home/alvasile/repos/shark-ai/sharktank/denseffnmoe/"
mlir_path = cwd + "denseffnmoe.mlir"
iree_module_path = cwd + "denseffnmoe.vmfb"
iree_result_path = cwd + "iree_result.npy"

model = ScatterLayer(Theta([DefaultPrimitiveTensor(data=torch.tensor([1]))]))

h_path = cwd + "h.npy"
expert_gate_path = cwd + "expert_gate.npy"
top_experts_index_path = cwd + "top_experts_index.npy"
h = torch.tensor(np.load(h_path))
expert_gate = torch.tensor(np.load(expert_gate_path))
top_experts_index = torch.tensor(np.load(top_experts_index_path))
expected_output = torch.tensor(np.load(cwd + "routed_out_slice.npy"))

# Run locally
eager_result = model.forward(h, top_experts_index, expert_gate)
assert (
    expected_output == eager_result
).all(), "Eager result does not match expected output"

num_tokens = torch.export.Dim("num_tokens")
input_feature_dim = torch.export.Dim("input_feature_dim")
num_top_experts = torch.export.Dim("num_top_experts")

dynamic_shapes = {
    "h": {0: num_tokens, 1: input_feature_dim},
    "top_experts_index": {0: num_tokens, 1: num_top_experts},
    "expert_gate": {0: num_tokens, 1: num_top_experts},
}

# Run through IREE
fxb = FxProgramsBuilder(model)


@fxb.export_program(
    name="scatter",
    args=(h, top_experts_index, expert_gate),
    # dynamic_shapes=dynamic_shapes,
    strict=False,
)
def _(model, h, top_experts_index, expert_gate) -> torch.Tensor:
    return model(h, top_experts_index, expert_gate)


output = export(fxb, import_symbolic_shape_expressions=True)
output.save_mlir(mlir_path)

compile_args = [
    f"iree-compile",
    f"{mlir_path}",
    f"-o={iree_module_path}",
    "--iree-opt-level=O3",
    "--iree-hal-target-device=hip[0]",
    "--iree-hip-target=gfx942",
]
cmd = subprocess.list2cmdline(compile_args)
print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeCompileException(proc, cwd)

# Write run command
run_args = [
    "iree-run-module",
    "--hip_use_streams=true",
    f"--module={iree_module_path}",
    "--device=hip://0",
    "--function=scatter",
    f"--input=@{h_path}",
    f"--input=@{top_experts_index_path}",
    f"--input=@{expert_gate_path}",
    f"--output=@{iree_result_path}",
]
cmd = subprocess.list2cmdline(run_args)
print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeBenchmarkException(proc, cwd)

iree_result = torch.tensor(np.load(iree_result_path))
assert (
    expected_output == iree_result
).all(), "IREE result does not match expected output"
