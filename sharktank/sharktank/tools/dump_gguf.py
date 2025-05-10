# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import re
import logging

import numpy as np

from sharktank.layers import *
from sharktank.types import *
from sharktank.utils import cli

logger = logging.getLogger(__name__)


def main():

    # Set up logging

    parser = cli.create_parser()
    parser.add_argument(
        "--dump-tensor-dir", type=Path, help="Dump tensor contents to a directory"
    )
    parser.add_argument(
        "--tensor-regex", type=str, help="Only dumps tensors matching a regex"
    )
    parser.add_argument(
        "--output-irpa", type=Path, help="Save the GGUF dataset to an IRPA file"
    )
    parser.add_argument(
        "--num-blocks", type=int, help="Number of tensors to save to an IRPA file"
    )
    parser.add_argument(
        "--save-input-output-blocks",
        action="store_true",
        help="Save input and output tensors",
    )

    cli.add_input_dataset_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser)
    config = cli.get_input_dataset(args)

    logger.setLevel(args.loglevel)

    model_arch = config.properties.get("general.architecture", "llama")
    if args.num_blocks:
        config.properties[f"{model_arch}.block_count"] = args.num_blocks
        num_blocks = list(range(0, args.num_blocks))
        logger.info(f"  Saving {num_blocks} blocks")
    else:
        num_blocks = range(0, config.properties[f"{model_arch}.block_count"])

    logger.info("  Properties:")
    for key, value in config.properties.items():
        logger.info(f"  {key} = {value} (of {type(value)})")

    tensors = []
    logger.info("  Tensors:")
    for tensor in config.root_theta.flatten().values():
        save = False
        # Save tensors of if name in tensor_regex
        if args.tensor_regex is not None and re.search(args.tensor_regex, tensor.name):
            save = True

        # Save input/output layer tensors
        if "blk" not in tensor.name and args.save_input_output_blocks:
            save = True
        elif int(tensor.name.split(".")[1]) in num_blocks:
            # Save tensors if in num_blocks
            save = True

        if save:
            logger.info(f"  {tensor.name}: {tensor.shape}")
            tensors += [
                DefaultPrimitiveTensor(data=tensor.as_torch(), name=tensor.name)
            ]

        if isinstance(tensor, PrimitiveTensor):
            torch_tensor = tensor.as_torch()
            logger.debug(
                f"    : torch.Tensor({list(torch_tensor.shape)}, "
                f"dtype={torch_tensor.dtype})"
            )
        elif isinstance(tensor, QuantizedTensor):
            logger.debug(f"    : QuantizedTensor({tensor.layout_type.__name__})")
            try:
                unpacked = tensor.unpack()
                logger.debug(f"    {unpacked}")
            except NotImplementedError:
                logger.warning(f"    Unpacking NOT IMPLEMENTED for {tensor.name}")
        elif isinstance(tensor, ShardedTensor):
            for i, pt in enumerate(tensor.shards):
                logger.debug(f"    {i}: {pt}")

        _maybe_dump_tensor(args, tensor)

    theta = Theta(tensors)
    props = config.properties
    dataset = Dataset(props, theta)

    if args.output_irpa:
        dataset.save(args.output_irpa, io_report_callback=logger.debug)


def _maybe_dump_tensor(args, t: InferenceTensor):
    if not args.dump_tensor_dir:
        return
    dir: Path = args.dump_tensor_dir
    dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping tensor {t.name} to {dir}")

    try:
        if isinstance(t, PrimitiveTensor):
            torch_tensor = t.as_torch()
            np.save(dir / f"{t.name}.npy", torch_tensor.detach().numpy())
        elif isinstance(t, QuantizedTensor):
            layout: QuantizedLayout = t.unpack()
            dq = layout.dequant()
            np.save(dir / f"{t.name}.dequant.npy", dq.detach().numpy())
        else:
            logger.error(f"Unexpected tensor type: {type(t)}")
            raise AssertionError(f"Unexpected tensor type: {type(t)}")
    except Exception as e:
        logger.error(f"Failed to dump tensor {t.name}: {str(e)}")


if __name__ == "__main__":
    main()
