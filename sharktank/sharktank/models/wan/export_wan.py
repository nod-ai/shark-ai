import os
from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

from sharktank.models.wan.export import export_wan_transformer_from_huggingface, wan_transformer_default_batch_sizes

def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(
        description="Export wan2.1 transformer MLIR from a parameters file."
    )
    parser.add_argument("--repo_id", type=str,
                        default="Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--mlir_path", type=str,
                        default=f"wan_transformer.mlir")
    parser.add_argument("--params_path", type=str,
                        default=f"wan_transformer_dataset.irpa")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=wan_transformer_default_batch_sizes,
    )
    args = parser.parse_args(args=args)
    export_wan_transformer_from_huggingface(
        repo_id=args.repo_id,
        mlir_output_path=args.mlir_path,
        parameters_output_path=args.params_path,
        batch_sizes=args.batch_sizes,
    )
    print("export_wan_transformer_from_huggingface done")

if __name__ == "__main__":
    main()
