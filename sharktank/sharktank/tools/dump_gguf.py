# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import re
import logging

import numpy as np
import torch

from ..layers import *
from ..types import *

logger = logging.getLogger(__name__)


def main():
    from ..utils import cli

    # Set up logging

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--dump-tensor-dir", type=Path, help="Dump tensor contents to a directory"
    )
    parser.add_argument(
        "--tensor-regex", type=str, help="Only dumps tensors matching a regex"
    )
    parser.add_argument(
        "--save", type=Path, help="Save the GGUF dataset to an IRPA file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = cli.parse(parser)

    # Configure logging based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    config = cli.get_input_dataset(args)

    if args.save is not None:

        def report(s):
            logger.info(f"Save: {s}")

        logger.info(f"Saving to: {args.save}")
        config.save(args.save, io_report_callback=report)
        return

    logger.debug("Properties:")
    for key, value in config.properties.items():
        logger.debug(f"  {key} = {value} (of {type(value)})")

    logger.debug("Tensors:")
    for tensor in config.root_theta.flatten().values():
        if args.tensor_regex is not None:
            if not re.search(args.tensor_regex, tensor.name):
                continue

        logger.debug(f"  {tensor}")
        if isinstance(tensor, PrimitiveTensor):
            torch_tensor = tensor.as_torch()
            logger.debug(
                f"    : torch.Tensor({list(torch_tensor.shape)}, "
                f"dtype={torch_tensor.dtype}) = {tensor.as_torch()}"
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
