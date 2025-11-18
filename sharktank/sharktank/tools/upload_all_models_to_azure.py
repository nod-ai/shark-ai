# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.utils.azure import upload_all_models
from amdsharktank.utils.logging import get_logger

import logging
import argparse


def main(args: list[str] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Upload all models to Azure storage. Uploads only if files are different. "
            "If they need updating a snapshot will be created before uploading."
        )
    )
    parser.add_argument(
        "--account-name", type=str, required=True, help="Storage account name."
    )
    parser.add_argument("--container-name", type=str, required=True)
    parser.add_argument(
        "--account-key",
        type=str,
        default=None,
        help=(
            "Access key. If not provided, will use environment variable AZURE_STORAGE_KEY"
            " as key. If this is not available, will use the default Azure credential."
        ),
    )
    parser.add_argument(
        "--destination-name-prefix",
        type=str,
        required=True,
        help="Name prefix of all blobs that will be uploaded.",
    )
    parsed_args = parser.parse_args(args)

    upload_all_models(
        account_name=parsed_args.account_name,
        container_name=parsed_args.container_name,
        destination_name_prefix=parsed_args.destination_name_prefix,
        account_key=parsed_args.account_key,
    )


if __name__ == "__main__":
    # Set the logging level for all azure-storage-* libraries
    azure_logger = logging.getLogger("azure.storage")
    azure_logger.setLevel(logging.INFO)

    main()
