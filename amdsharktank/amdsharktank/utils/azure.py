# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path
from amdsharktank.utils.logging import get_logger
from tempfile import TemporaryDirectory
from typing import Callable, Optional
import hashlib
import os


logger = get_logger(__name__)


def calculate_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.digest()


def create_blob_service_client(
    account_name: str, account_key: Optional[str] = None
) -> BlobServiceClient:
    if account_key is None and "AZURE_STORAGE_KEY" in os.environ:
        account_key = os.environ["AZURE_STORAGE_KEY"]
    if account_key:
        connection_string = (
            f"DefaultEndpointsProtocol=https;AccountName={account_name};"
            f"AccountKey={account_key};"
            "EndpointSuffix=core.windows.net"
        )
        return BlobServiceClient.from_connection_string(connection_string)

    credential = DefaultAzureCredential()
    account_url = f"https://{account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url, credential)


def snapshot_and_upload_blob_if_different(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_name: str,
    file_path: str,
):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    local_hash = calculate_hash(file_path)

    blob_exists = False
    try:
        blob_properties = blob_client.get_blob_properties()
        existing_hash = blob_properties.content_settings.content_md5
        blob_exists = True
    except Exception:
        existing_hash = None

    if local_hash == existing_hash:
        logger.info(f'Skipping upload to blob "{blob_name}".')
        return

    if blob_exists:
        blob_client.create_snapshot()

    with open(file_path, "rb") as f:
        logger.info(f'Uploading to blob "{blob_name}"...')
        content_settings = ContentSettings(content_md5=local_hash)
        blob_client.upload_blob(f, overwrite=True, content_settings=content_settings)
        logger.info(f'Blob "{blob_name}" uploaded.')


def upload_directory(
    blob_service_client: BlobServiceClient,
    container_name: str,
    source_dir: str,
    destination_blob_name_prefix: str,
):
    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            file_path = Path(root) / file_name
            blob_name = f"{destination_blob_name_prefix}{os.path.relpath(file_path, source_dir)}"
            snapshot_and_upload_blob_if_different(
                blob_service_client, container_name, blob_name, file_path
            )


def upload_model(
    export_fn: Callable[[Path], None],
    blob_service_client: BlobServiceClient,
    container_name: str,
    destination_blob_name_prefix: str,
):
    with TemporaryDirectory() as tmp_dir:
        export_fn(Path(tmp_dir))
        upload_directory(
            blob_service_client,
            container_name,
            source_dir=tmp_dir,
            destination_blob_name_prefix=destination_blob_name_prefix,
        )


def upload_all_models(
    account_name: str,
    container_name: str,
    destination_name_prefix: str,
    account_key: Optional[str] = None,
):
    """Upload all models to Azure.
    Will generate temporary export artifacts.
    If MD5 hashes match with the existing blobs nothing will be uploaded.
    Creates snapshots if files need updating."""
    from amdsharktank.models.flux.export import export_flux_transformer_models

    blob_service_client = create_blob_service_client(account_name, account_key)

    upload_model(
        export_flux_transformer_models,
        blob_service_client,
        container_name,
        destination_name_prefix,
    )
    # TODO: add more models here
