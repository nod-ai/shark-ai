import sys
from typing import List
import argparse
import torch
from torch import Tensor
from amdsharktank import ops
from iree.turbine import aot


def export_gemm(
    mlir_path: str,
    device_count: int,
    m: int,
    n: int,
    k: int,
    with_alpha: bool,
    with_beta: bool,
):
    class GemmModule(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return ops.gemm(*args, **kwargs)

    a = torch.empty(m, k, dtype=torch.float32)
    b = torch.empty(k, n, dtype=torch.float32)
    c = torch.empty(m, n, dtype=torch.float32)
    sharded_a = ops.reshard_split(a, dim=0, count=device_count)
    sharded_b = ops.replicate(b, count=device_count)
    sharded_c = ops.reshard_split(c, dim=0, count=device_count)
    gemm_module = GemmModule()
    kwargs = {
        "a": sharded_a,
        "b": sharded_b,
        "c": sharded_c,
    }
    # Need to pass alpha and beta not as numbers, but as tensors since
    # the IREE FX importer does not support ConstantArgument.
    if with_alpha:
        kwargs["alpha"] = torch.tensor(2.0, dtype=torch.float32)
    if with_alpha:
        kwargs["beta"] = torch.tensor(3.0, dtype=torch.float32)
    torch_exported = torch.export.export(gemm_module, args=(), kwargs=kwargs)
    export_output = aot.export(torch_exported)
    export_output.save_mlir(mlir_path)


def export_gemm_cli(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="""
Export sharded GEMM to MLIR.
alpha * a @ b + beta * c
a is MxK matrix.
b is KxN matrix.
c is MxN matrix.
The sharded/split dimension is M.
a and c will be split across dimension 0 (M).
b will be replicated on all devices.
For n devices the exported function will have signature
(a0, a1, ..., an, b0, b1, ..., bn, c0, c1, ..., cn) -> (r0, r1, ..., rn),
where ai and ci are the respective shards on the i-th device.
bi is equal to b, but on the i-th device.
The caller must place the shards on the expected devices.

The result is split along dimension M also,
where ri is on the i-th device.

Support for --with-alpha and --with-beta is under construction.

Example usage:
python export_gemm.py --device_count=2 --m=10, --k=20, --n=30 \\
    --mlir=sharded-gemm.mlir""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mlir", help="Path to the exported program.", type=str, required=True
    )
    parser.add_argument(
        "--device_count", help="Number of shards/devices", type=int, required=True
    )
    parser.add_argument("--m", help="M", type=int, default=512)
    parser.add_argument("--n", help="N", type=int, default=512)
    parser.add_argument("--k", help="K", type=int, default=512)
    parser.add_argument(
        "--with-alpha",
        help="Have alpha as an argument to the function signature",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--with-beta",
        help="Have alpha as an argument to the function signature",
        default=False,
        action="store_true",
    )
    args = parser.parse_args(args=argv[1:])
    export_gemm(
        mlir_path=args.mlir,
        device_count=args.device_count,
        m=args.m,
        n=args.n,
        k=args.k,
        with_alpha=args.with_alpha,
        with_beta=args.with_beta,
    )


if __name__ == "__main__":
    export_gemm_cli(sys.argv)
