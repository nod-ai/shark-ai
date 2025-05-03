# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import multiprocessing as mp
from queue import Empty
from threading import Timer
import time
from typing import Dict, Any, Optional
import json
import os

import uvicorn.logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from shortfin.support.responder import AbstractResponder
from .components.generate import ClientGenerateBatchProcess

from .components.lifecycle import ShortfinLlmLifecycleManager

from .components.io_struct import GenerateReqInput, SamplingParams

import uvicorn

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "[{asctime}] {message}",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{",
            "use_colors": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# class SimulatedLLMService:
#     """Simulated LLM service that mimics the behavior of ShortfinLlmLifecycleManager"""
#     def __init__(self, args):
#         self.args = args
#         # Load model config
#         with open(args.model_config, 'r') as f:
#             self.model_config = json.load(f)
#         logger.info("Initialized simulated LLM service")

#     def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Simulate LLM generation"""
#         prompt = request_data.get('text', '')  # Changed from 'prompt' to 'text' to match input
#         max_tokens = request_data.get('sampling_params', {}).get('max_completion_tokens', 100)

#         logger.debug(f"Processing request with prompt: {prompt}")
#         # Simulate some processing time
#         time.sleep(0.1)

#         # Return a simulated response
#         response = {
#             "text": f"Simulated response to: {prompt}",
#             "tokens": list(range(max_tokens)),
#             "finish_reason": "length"
#         }
#         logger.debug(f"Generated response: {response}")
#         return response


class RequestQueueManager:
    def __init__(self):
        # Use Manager to create shared queues
        self.manager = mp.Manager()
        self.request_queue = self.manager.Queue()
        self.response_queues = (
            self.manager.dict()
        )  # Shared dictionary for response queues
        logger.debug("Initialized RequestQueueManager")

    def submit_request(self, request_id: str, request_data: Dict[str, Any]) -> mp.Queue:
        """Submit a request and return the response queue for that request"""
        logger.debug(f"Submitting request {request_id}: {request_data}")
        response_queue = self.manager.Queue()  # Create a new shared queue
        self.response_queues[request_id] = response_queue
        self.request_queue.put((request_id, request_data))
        return response_queue

    def get_request(self):
        """Get the next request from the queue"""
        logger.debug("Waiting for request from queue")
        request = self.request_queue.get()
        logger.debug(f"Got request from queue: {request}")
        return request

    def put_response(self, request_id: str, response_data: Dict[str, Any]):
        """Put a response in the appropriate response queue"""
        logger.debug(f"Putting response for request {request_id}: {response_data}")
        if request_id in self.response_queues:
            self.response_queues[request_id].put(response_data)
            del self.response_queues[request_id]
            logger.debug(f"Response sent for request {request_id}")
        else:
            logger.error(f"No response queue found for request {request_id}")


class CliResponder(AbstractResponder):
    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self.response = asyncio.Future(loop=self._loop)
        self.responded = False
        self.timer = Timer()

    def start_response(self):
        self.timer.start()

    def ensure_response(self):
        self.timer.end()

    def send_response(self, response):
        assert not self.responded, "Response already sent"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self.responded = True
        self._loop.call_soon_threadsafe(self.response.set_result, response)

    def stream_start(self, **kwargs):
        raise Exception("Streaming not supported")

    def stream_part(self, content):
        raise Exception("Streaming not supported")


def lifecycle_manager_process(queue_manager: RequestQueueManager, args):
    """Process that runs the simulated LLM service"""
    logger.info("Starting simulated LLM service process")
    lifecycle_manager = ShortfinLlmLifecycleManager(args)
    service = lifecycle_manager.services["default"]
    service.start()

    while True:
        try:
            request_id, request_data = queue_manager.get_request()
            gen_req = GenerateReqInput(
                text=request_data.get("text", ""),
                sampling_params=SamplingParams(
                    max_completion_tokens=request_data.get("sampling_params", {}).get(
                        "max_completion_tokens", 100
                    )
                ),
            )
            try:
                # Process the request
                # response = llm_service.generate(request_data)
                responder = CliResponder()
                ClientGenerateBatchProcess(service, gen_req, responder).launch()
                asyncio.run(responder.response)
                response = responder.response.result()
                queue_manager.put_response(request_id, response)
            except Exception as e:
                logger.error(f"Error processing request {request_id}: {str(e)}")
                queue_manager.put_response(request_id, {"error": str(e)})
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
            break
        except Exception as e:
            logger.error(f"Error in LLM service process: {str(e)}")
            time.sleep(1)  # Prevent tight loop on errors


def create_fastapi_app(queue_manager: RequestQueueManager) -> FastAPI:
    """Create the FastAPI app that handles HTTP requests"""
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        request_data = await request.json()
        request_id = str(id(request))  # Simple unique ID for the request
        logger.debug(f"Received request {request_id}: {request_data}")

        # Submit request to LLM service
        response_queue = queue_manager.submit_request(request_id, request_data)

        try:
            # Wait for response with timeout
            logger.debug(f"Waiting for response for request {request_id}")
            response = response_queue.get(timeout=30)  # 30 second timeout
            logger.debug(f"Got response for request {request_id}: {response}")
            return JSONResponse(content=response)
        except Empty:
            logger.error(f"Request {request_id} timed out")
            return JSONResponse(status_code=504, content={"error": "Request timed out"})
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_json", type=Path, required=True)
    parser.add_argument("--tokenizer_config_json", type=Path, required=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--token_selection_strategy", type=str, default="greedy")
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--vmfb", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, nargs="*")
    parser.add_argument("--device", type=str, default="hip")
    parser.add_argument("--device_ids", type=int, nargs="*", default=[0])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-keep-alive", type=int, default=5)
    args = parser.parse_args(argv)
    if args.tokenizer_config_json is None:
        # this is only used for the EOS token
        logging.info("Argument `--tokenizer_config_json` is not provided")
        logging.info("Inferring tokenizer config path from tokenizer path")
        inferred_tokenizer_config_path = args.tokenizer_json.with_name(
            args.tokenizer_json.stem + "_config.json"
        )
        args.tokenizer_config_json = inferred_tokenizer_config_path

    return args


def run_server(argv, log_config=uvicorn.config.LOGGING_CONFIG, port: int | None = None):
    args = parse_args(argv)

    # Create queue manager for inter-process communication
    queue_manager = RequestQueueManager()

    # Start LLM service process
    llm_proc = mp.Process(target=lifecycle_manager_process, args=(queue_manager, args))
    llm_proc.start()
    logger.info("Started LLM service process")

    try:
        # Create and run FastAPI app
        app = create_fastapi_app(queue_manager)
        logger.info(f"Starting FastAPI server on port {port or args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=port or args.port,
            log_config=log_config,
            timeout_keep_alive=args.timeout_keep_alive,
        )
    finally:
        # Cleanup
        logger.info("Shutting down LLM service process")
        llm_proc.terminate()
        llm_proc.join()


if __name__ == "__main__":
    run_server(
        sys.argv[1:],
        log_config=UVICORN_LOG_CONFIG,
    )
