# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime as dt
import time
import json
import argparse
import asyncio
import aiohttp
import requests

from shortfin_apps.types.Base64CharacterEncodedByteSequence import (
    Base64CharacterEncodedByteSequence,
)

from shortfin_apps.utilities.image import (
    save_to_file,
    image_from,
)

sample_request = {
    "prompt": [
        " a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
    ],
    "neg_prompt": ["Watermark, blurry, oversaturated, low resolution, pollution"],
    "height": [1024],
    "width": [1024],
    "steps": [20],
    "guidance_scale": [7.5],
    "seed": [0],
    "output_type": ["base64"],
    "rid": ["string"],
}


async def send_request(
    session: aiohttp.ClientSession,
    rep: int,
    args: argparse.Namespace,
    data: dict,
):
    print("Sending request batch #", rep)
    url = f"{args.host}:{args.port}/generate"
    start = time.time()
    async with session.post(url, json=data) as response:
        end = time.time()
        # Check if the response was successful
        if response.status == 200:
            response.raise_for_status()  # Raise an error for bad responses
            res_json = await response.json(content_type=None)
            if args.save:
                for idx, each_png in enumerate(res_json["images"]):
                    if not isinstance(each_png, str):
                        raise ValueError(f"png was not string at index {idx}")

                    each_image = image_from(
                        Base64CharacterEncodedByteSequence(each_png)
                    )

                    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
                    each_file_name = f"shortfin_sd_output_{timestamp}_{idx}.png"

                    each_file_path = save_to_file(
                        each_image, args.outputdir, each_file_name
                    )

                    print(f"Saved to {each_file_path}")
            latency = end - start
            print("Responses processed.")
            return latency, len(data["prompt"])
        print(f"Error: Received {response.status} from server")
        raise Exception


async def static(args):
    # Create an aiohttp session for sending requests
    async with aiohttp.ClientSession() as session:
        pending = []
        latencies = []
        sample_counts = []
        # Read the JSON file if supplied. Otherwise, get user input.
        try:
            if not args.file:
                data = sample_request
            else:
                with open(args.file, "r") as json_file:
                    data = json.load(json_file)
        except Exception as e:
            print(f"Error reading the JSON file: {e}")
            return
        data["prompt"] = (
            [data["prompt"]] if isinstance(data["prompt"], str) else data["prompt"]
        )
        start = time.time()

        async for i in async_range(args.reps):
            pending.append(asyncio.create_task(send_request(session, i, args, data)))
            await asyncio.sleep(1)  # Wait for 1 second before sending the next request
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.ALL_COMPLETED
            )
            for task in done:
                latency, num_samples = await task
                latencies.append(latency)
                sample_counts.append(num_samples)
        end = time.time()
        if not any(i is None for i in [latencies, sample_counts]):
            total_num_samples = sum(sample_counts)
            sps = str(total_num_samples / (end - start))
            # Until we have better measurements, don't report the throughput that includes saving images.
            if not args.save:
                print(f"Average throughput: {sps} samples per second")
        else:
            raise ValueError("Received error response from server.")


async def interactive(args):
    # Create an aiohttp session for sending requests
    async with aiohttp.ClientSession() as session:
        pending = []
        latencies = []
        sample_counts = []
        # Read the JSON file if supplied. Otherwise, get user input.
        try:
            if not args.file:
                data = sample_request
            else:
                with open(args.file, "r") as json_file:
                    data = json.load(json_file)
        except Exception as e:
            print(f"Error reading the JSON file: {e}")
            return
        data["prompt"] = (
            [data["prompt"]] if isinstance(data["prompt"], str) else data["prompt"]
        )
        while True:
            prompt = await ainput("Enter a prompt: ")
            data["prompt"] = [prompt]
            data["steps"] = [args.steps]
            print("Sending request with prompt: ", data["prompt"])

            async for i in async_range(args.reps):
                pending.append(
                    asyncio.create_task(send_request(session, i, args, data))
                )
                await asyncio.sleep(
                    1
                )  # Wait for 1 second before sending the next request
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.ALL_COMPLETED
                )
                for task in done:
                    _, _ = await task
            pending = []
            if any(i is None for i in [latencies, sample_counts]):
                raise ValueError("Received error response from server.")


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, f"{prompt} ")


async def async_range(count):
    for i in range(count):
        yield i
        await asyncio.sleep(0.0)


def is_connected(host: str, port: int) -> bool:
    max_port = 65535
    if port < 1 or port > max_port:
        print(
            f"Error: Invalid port specified ({port}), expected a value between 1 and {max_port}"
        )
        return False

    url = f"{host}:{port}"
    ready = False
    print("Waiting for server.", end=None)
    while not ready:
        try:
            if requests.get(f"{url}/health", timeout=20).status_code == 200:
                print("Successfully connected to server.")
                ready = True
                break
            time.sleep(2)
            print(".", end=None)
        except:
            time.sleep(2)
            print(".", end=None)

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="A non-default request to send to the server.",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of times to duplicate each request in one second intervals.",
    )
    p.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save images. To disable, use --no-save",
    )
    p.add_argument(
        "--outputdir",
        type=str,
        default="gen_imgs",
        help="Directory to which images get saved.",
    )
    p.add_argument(
        "--host", type=str, default="http://0.0.0.0", help="Server host address."
    )
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument(
        "--steps",
        type=int,
        default="20",
        help="Number of inference steps. More steps usually means a better image. Interactive only.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Start as an example CLI client instead of sending static requests.",
    )
    args = p.parse_args()

    if not is_connected(args.host, args.port):
        exit(3)

    if args.interactive:
        asyncio.run(interactive(args))
    else:
        asyncio.run(static(args))


if __name__ == "__main__":
    main()
