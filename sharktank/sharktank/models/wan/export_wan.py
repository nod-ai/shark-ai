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
                        default="wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--mlir_path", type=str,
                        default=f"wan_transformer_bf16.mlir")
    parser.add_argument("--params_path", type=str,
                        default=f"wan_transformer_dataset_bf16.irpa")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=wan_transformer_default_batch_sizes,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81
    )
    args = parser.parse_args(args=args)
    export_wan_transformer_from_huggingface(
        repo_id=args.repo_id,
        mlir_output_path=args.mlir_path,
        parameters_output_path=args.params_path,
        batch_sizes=args.batch_sizes,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames
    )
    print("export_wan_transformer_from_huggingface done")

if __name__ == "__main__":
    main()
