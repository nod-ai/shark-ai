# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests
import time
import asyncio
import pytest
import subprocess
import os
import socket
import sys
import copy
import math
import tempfile
import huggingface_hub
from contextlib import closing
from pathlib import Path

from datetime import datetime as dt
from PIL.Image import Image

from shortfin_apps.types.Base64CharacterEncodedByteSequence import (
    Base64CharacterEncodedByteSequence,
)

from shortfin_apps.utilities.image import (
    image_from,
)

BATCH_SIZES = [1]

sample_request = {
    "prompt": [
        " a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
    ],
    "neg_prompt": ["blurry, low quality, simplified"],
    "height": [1024],
    "width": [1024],
    "steps": [2],
    "guidance_scale": [7.5],
    "seed": [42],
    "output_type": ["base64"],
    "rid": ["string"],
}


def start_server(pipeline_variant: str, fibers_per_device=1, isolation="per_fiber"):
    # Start the server
    model_config_path = model_config[pipeline_variant]
    srv_args = [
        "python",
        "-m",
        "shortfin_apps.flux.server",
        f"--model_config={model_config_path}",
        f"--fibers_per_device={fibers_per_device}",
        f"--isolation={isolation}",
    ]
    runner = ServerRunner(srv_args)
    # Wait for server to start
    time.sleep(3)
    return runner


model_config = {
    "flux_dev": str(
        Path(__file__).parent.parent.parent.parent
        / "python"
        / "shortfin_apps"
        / "flux"
        / "examples"
        / "flux_dev_config.json"
    ),
    "flux_schnell": str(
        Path(__file__).parent.parent.parent.parent
        / "python"
        / "shortfin_apps"
        / "flux"
        / "examples"
        / "flux_schnell_config.json"
    ),
}

hf_repo = {
    "flux_dev": "black-forest-labs/FLUX.1-dev",
    "flux_schnell": "black-forest-labs/FLUX.1-schnell",
}


@pytest.fixture(scope="module", params=["flux_dev", "flux_schnell"])
def pipeline_variant(request):
    yield request.param


@pytest.fixture(scope="module")
def pipeline_parameters(pipeline_variant):
    """Download and export model parameters."""
    export_script_path = str(
        Path(__file__).parent.parent.parent.parent.parent
        / "amdsharktank"
        / "amdsharktank"
        / "pipelines"
        / "flux"
        / "export_from_hf.sh"
    )
    hf_model_dir = huggingface_hub.snapshot_download(hf_repo[pipeline_variant])
    subprocess.check_call([export_script_path, hf_model_dir, pipeline_variant])


@pytest.fixture(scope="module")
def flux_server_fpd1(pipeline_variant, pipeline_parameters):
    runner = start_server(pipeline_variant=pipeline_variant, fibers_per_device=1)
    yield runner
    # Teardown: kill the server
    del runner


@pytest.fixture(scope="module")
def flux_server_fpd1_per_call(pipeline_variant, pipeline_parameters):
    runner = start_server(
        pipeline_variant=pipeline_variant, fibers_per_device=1, isolation="per_call"
    )
    yield runner
    # Teardown: kill the server
    del runner


@pytest.fixture(scope="module")
def flux_server_fpd2(pipeline_variant, pipeline_parameters):
    runner = start_server(pipeline_variant=pipeline_variant, fibers_per_device=2)
    yield runner
    # Teardown: kill the server
    del runner


@pytest.mark.system("hip")
def test_smoke_flux_server(flux_server_fpd1):
    imgs, status_code = send_json_file(flux_server_fpd1.url)
    assert len(imgs) == 1
    assert status_code == 200


@pytest.mark.system("hip")
def test_smoke_flux_server_bs4_dense(flux_server_fpd1):
    imgs, status_code = send_json_file(flux_server_fpd1.url, num_copies=4)
    assert len(imgs) == 4
    assert status_code == 200


@pytest.mark.system("hip")
def test_smoke_flux_server_bs4_dense_fpd2(flux_server_fpd2):
    imgs, status_code = send_json_file(flux_server_fpd2.url, num_copies=4)
    assert len(imgs) == 4
    assert status_code == 200


class ServerRunner:
    def __init__(self, args):
        port = str(find_free_port())
        self.url = "http://0.0.0.0:" + port
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            [
                *args,
                "--port=" + port,
                "--device=hip",
            ],
            env=env,
            # TODO: Have a more robust way of forking a subprocess.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        start = time.time()
        while True:
            time.sleep(2)
            try:
                if requests.get(f"{self.url}/health").status_code == 200:
                    return
            except Exception as e:
                if self.process.errors is not None:
                    raise RuntimeError("API server process terminated") from e
            time.sleep(1.0)
            if (time.time() - start) > 30:
                raise RuntimeError("Timeout waiting for server start")

    def __del__(self):
        try:
            process = self.process
        except AttributeError:
            pass
        else:
            process.terminate()
            process.wait()


def send_json_file(url="http://0.0.0.0:8000", num_copies=1):
    # Read the JSON file
    data = copy.deepcopy(sample_request)
    imgs: list[Image] = []
    # Send the data to the /generate endpoint
    data["prompt"] = (
        [data["prompt"]]
        if isinstance(data["prompt"], str)
        else data["prompt"] * num_copies
    )
    try:
        response = requests.post(url + "/generate", json=data)
        response.raise_for_status()  # Raise an error for bad responses
        response_body = response.json()

        for idx, each_png in enumerate(response_body["images"]):
            if not isinstance(each_png, str):
                raise ValueError(
                    f"Expected string-encoded png at index {idx}, found {each_png}"
                )
            each_image = image_from(Base64CharacterEncodedByteSequence(each_png))
            imgs.append(each_image)

    except requests.exceptions.RequestException as e:
        print(f"Error sending the request: {e}")

    return imgs, response.status_code


def find_free_port():
    """This tries to find a free port to run a server on for the test.

    Race conditions are possible - the port can be acquired between when this
    runs and when the server starts.

    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def test_placeholder():
    # Here in case this pytest is invoked via CPU CI and no tests are run.
    pass
