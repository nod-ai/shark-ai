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

    def forward(self):
        out = torch.zeros([10, 10], dtype=torch.float32)
        dim = 0
        src = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        index = torch.stack([torch.arange(10) for _ in range(10)])
        out.scatter_(dim, index, src)
        return out


mlir_path = "/home/alvasile/repos/shark-ai/sharktank/scatter_.mlir"
iree_module_path = "/home/alvasile/repos/shark-ai/sharktank/scatter_.vmfb"
iree_result_path = "/home/alvasile/repos/shark-ai/sharktank/scatter__iree_result.npy"

model = ScatterLayer(Theta([DefaultPrimitiveTensor(data=torch.tensor([1]))]))
fxb = FxProgramsBuilder(model)


@fxb.export_program(
    name="scatter_",
    args=(),
    strict=False,
)
def _(model) -> torch.Tensor:
    return model()


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
cwd = "/home/alvasile/repos/shark-ai/sharktank"
# logger.info(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeCompileException(proc, cwd)

# Write run command
run_args = [
    "iree-run-module",
    "--hip_use_streams=true" f"--module={iree_module_path}",
    "--device=hip://1",
    # f"--output={iree_result_path}",
    "--function=scatter_",
]
cmd = subprocess.list2cmdline(run_args)
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
# if return_code != 0:
# raise IreeBenchmarkException(proc, cwd)
# Load from saved numpy
# iree_result = np.load(iree_result_path)
# Run locally
eager_result = model.forward()
# Compare
print(eager_result)
