import subprocess
import torch
import numpy as np

import torch.nn.functional as F
from iree.turbine.aot import *
from sharktank.layers.base import ThetaLayer
from sharktank import ops
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

        # (self.num_experts, num_tokens)
        router_indices = torch.arange(num_tokens).view(1, -1).expand(num_experts, -1)
        # (self.num_experts * num_tokens, input_feature_dim)
        router_indices = (
            router_indices.reshape(-1, 1)
            .expand(-1, input_feature_dim)
            .to(dtype=torch.float)
        )
        return router_indices.view(num_experts, num_tokens, input_feature_dim).sum(
            dim=0
        )


import os

name = "fail"

cwd = os.path.realpath(os.curdir)
cwd = "/home/alvasile/repos/shark-ai/sharktank/denseffnmoe/"
mlir_path = cwd + f"{name}.mlir"
iree_module_path = cwd + f"{name}.vmfb"
iree_result_path = cwd + "iree_result.npy"

model = ScatterLayer(Theta([DefaultPrimitiveTensor(data=torch.tensor([1]))]))

ffn_input_path = cwd + "ffn_input.npy"
expert_gate_path = cwd + "expert_gate.npy"
top_experts_index_path = cwd + "top_k_experts.npy"
ffn_input = torch.tensor(np.load(ffn_input_path))
expert_gate = torch.tensor(np.load(expert_gate_path))
top_k_experts = torch.tensor(np.load(top_experts_index_path))
expected_output = torch.tensor(np.load(cwd + "eager_moe_output.npy"))

# Run locally
eager_result = model.forward(ffn_input, top_k_experts, expert_gate)
# assert torch.isclose(
# eager_result, expected_output, rtol=1.3e-6, atol=1e-5
# ).all(), "Eager result does not match expected output"


num_tokens = torch.export.Dim("num_tokens")
input_feature_dim = torch.export.Dim("input_feature_dim")
num_top_experts = torch.export.Dim("num_top_experts")

dynamic_shapes = {
    "ffn_input": {0: num_tokens},  # , 1: input_feature_dim},
    "top_k_experts": {},  # {0: num_tokens},  # , 1: num_top_experts},
    "expert_gate": {},  # {0: num_tokens},  # , 1: num_top_experts},
}

# Run through IREE
fxb = FxProgramsBuilder(model)


@fxb.export_program(
    name="scatter",
    args=(ffn_input, top_k_experts, expert_gate),
    dynamic_shapes=dynamic_shapes,
    strict=False,
)
def _(model, ffn_input, top_k_experts, expert_gate) -> torch.Tensor:
    return model(ffn_input, top_k_experts, expert_gate)


output = export(fxb, import_symbolic_shape_expressions=True)
output.save_mlir(mlir_path)

compile_args = [
    "iree-compile",
    mlir_path,
    f"-o={iree_module_path}",
    "--iree-hal-target-device=hip",
    "--iree-hip-target=gfx942",
    "--iree-opt-level=O3",
    "--iree-hal-indirect-command-buffers=true",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-hal-memoization=true",
]
cmd = subprocess.list2cmdline(compile_args)
# print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
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
    f"--input=@{ffn_input_path}",
    f"--input=@{top_experts_index_path}",
    f"--input=@{expert_gate_path}",
    f"--output=@{iree_result_path}",
]
cmd = subprocess.list2cmdline(run_args)
# print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeBenchmarkException(proc, cwd)

iree_result = torch.tensor(np.load(iree_result_path))
print(torch.isclose(iree_result, eager_result, rtol=1.3e-6, atol=1e-5).all().item())
