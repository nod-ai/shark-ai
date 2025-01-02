# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Union

import requests
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class AccuracyValidationException(RuntimeError):
    pass


def run_subprocess(cmd, **kwargs):
    """
    Run a subprocess command, combining stdout and stderr.
    Only output if there's an error.

    Args:
        cmd: Command to run (list or string)
        **kwargs: Additional arguments to pass to subprocess.Popen

    Returns:
        CompletedProcess instance

    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit status
    """
    # Ensure we always capture output and combine streams
    kwargs.update(
        {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "universal_newlines": True,
        }
    )

    try:
        # Run the process
        process = subprocess.Popen(cmd, **kwargs)

        # Capture output
        output, _ = process.communicate()

        # Check return code
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            logger.error(f"Command output:\n{output}")
            raise subprocess.CalledProcessError(process.returncode, cmd, output)

        return subprocess.CompletedProcess(cmd, process.returncode, output, None)

    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            raise
        logger.error(f"Failed to execute command: {e}")
        logger.error(f"Command was: {cmd}")
        raise


def download_huggingface_model(local_dir, repo_id, model_file):
    model_path = local_dir / model_file
    logger.info(f"Preparing model_path: {model_path}..")
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {repo_id} {model_file} from Hugging Face...")
        run_subprocess(
            f"huggingface-cli download --local-dir {local_dir} {repo_id} {model_file}",
            shell=True,
        )
        logger.info(f"Model downloaded to {model_path}")
    else:
        logger.info("Using cached model")


def download_with_hf_datasets(local_dir: Union[Path, str], model_name: str):
    """Download a model using `sharktank.utils.hf_datasets` script.

    Args:
        local_dir (Path | str): Local directory to download model to.
        model_name (str): Name of model to download.
    """
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)

    logger.info(f"Download model {model_name} with `hf_datasets` to {local_dir}...")
    run_subprocess(
        [
            "python",
            "-m",
            "sharktank.utils.hf_datasets",
            model_name,
            "--local-dir",
            local_dir,
        ]
    )
    logger.info(f"Model {model_name} successfully downloaded.")


def download_tokenizer(local_dir, tokenizer_id):
    # Set up tokenizer if it doesn't exist
    tokenizer_path = local_dir / "tokenizer.json"
    logger.info(f"Preparing tokenizer_path: {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
        logger.info(f"Downloading tokenizer {tokenizer_id} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
        )
        tokenizer.save_pretrained(local_dir)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    else:
        logger.info("Using cached tokenizer")


def export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes):
    bs_string = ",".join(map(str, batch_sizes))
    logger.info(
        "Exporting model with following settings:\n"
        f"  MLIR Path: {mlir_path}\n"
        f"  Config Path: {config_path}\n"
        f"  Batch Sizes: {bs_string}"
    )
    run_subprocess(
        [
            "python",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--block-seq-stride=16",
            f"--{model_path.suffix.strip('.')}-file={model_path}",
            f"--output-mlir={mlir_path}",
            f"--output-config={config_path}",
            f"--bs={bs_string}",
        ]
    )
    logger.info(f"Model successfully exported to {mlir_path}")


def compile_model(mlir_path, vmfb_path, device_settings):
    logger.info(f"Compiling model to {vmfb_path}")
    run_subprocess(
        [
            "iree-compile",
            mlir_path,
            "-o",
            vmfb_path,
        ]
        + device_settings["device_flags"]
    )
    logger.info(f"Model successfully compiled to {vmfb_path}")


def find_available_port():
    import socket
    from contextlib import closing

    logger.info(f"Finding available port...")
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        logger.info(f"Found available port: {port}")
        return port


def wait_for_server(url, timeout=10):
    logger.info(f"Waiting for server to start at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health")
            logger.info("Server successfully started")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout} seconds")


def _start_llm_server_args(
    tokenizer_path,
    model_config_path,
    vmfb_path,
    parameters_path,
    settings,
    port,
):
    return [
        sys.executable,
        "-m",
        "shortfin_apps.llm.server",
        f"--tokenizer_json={tokenizer_path}",
        f"--model_config={model_config_path}",
        f"--vmfb={vmfb_path}",
        f"--parameters={parameters_path}",
        f"--device={settings['device']}",
        f"--port={port}",
    ]


def start_llm_server(
    port,
    tokenizer_path,
    model_config_path,
    vmfb_path,
    parameters_path,
    settings,
    timeout=10,
    multi=False,
):
    logger.info("Starting LLM server...")
    cmd_args = _start_llm_server_args(
        tokenizer_path,
        model_config_path,
        vmfb_path,
        parameters_path,
        settings,
        port,
    )

    if multi:
        server_process = multiprocessing.Process(
            target=run_subprocess, args=(cmd_args,)
        )
        server_process.start()
    else:
        server_process = run_subprocess(cmd_args)

    logger.info("Process started... waiting for server")
    wait_for_server(f"http://localhost:{port}", timeout)
    return server_process


def start_log_group(headline):
    # check if we are in github ci
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return f"\n::group::{headline}"
    return ""


def end_log_group():
    # check if we are in github ci
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "\n::endgroup::"
    return ""
