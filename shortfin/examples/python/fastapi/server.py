# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
import traceback
from contextlib import asynccontextmanager
import json
import threading
import sys

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import shortfin as sf
from shortfin.interop.fastapi import FastAPIResponder
import uvicorn


class RequestMessage(sf.Message):
    def __init__(self, request_value: int, responder: FastAPIResponder):
        super().__init__()
        self.request_value = request_value
        self.responder = responder


class System:
    def __init__(self):
        self.ls = sf.host.CPUSystemBuilder().create_system()
        # TODO: Come up with an easier bootstrap thing than manually
        # running a thread.
        self.t = threading.Thread(target=lambda: self.ls.run(self.run()))
        self.request_queue = self.ls.create_queue("request")
        self.request_writer = self.request_queue.writer()

    def start(self):
        self.t.start()

    def shutdown(self):
        self.request_queue.close()

    async def run(self):
        print("*** Sytem Running ***")
        request_reader = self.request_queue.reader()
        while request := await request_reader():
            try:
                responder = request.responder
                if request.request_value == 0:
                    raise ValueError("Something broke")
                elif request.request_value > 20:
                    responder.send_response(Response(status_code=400))
                elif request.request_value == 1:
                    # Send a single response.
                    responder.send_response(
                        JSONResponse({"answer": request.request_value})
                    )
                elif request.request_value == 2:
                    count = 0
                    while not responder.is_disconnected() and count < 5:
                        count += 1
                        await asyncio.sleep(1)
                    if count >= 2:
                        # Simulating a long response that takes 5 seconds to complete.
                        # If not enough time has passed, the response will not be generated.
                        responder.send_response(
                            JSONResponse({"answer": request.request_value})
                        )
                else:
                    # Stream responses from 0..value
                    responder.stream_start()
                    for i in range(request.request_value + 1):
                        if responder.is_disconnected():
                            continue
                        responder.stream_part(
                            (json.dumps({"answer": i}) + "\n\0").encode()
                        )
                        await asyncio.sleep(0.01)
                    responder.stream_part(None)
            except:
                traceback.print_exc()
            finally:
                responder.ensure_response()


@asynccontextmanager
async def lifespan(app: FastAPI):
    system.start()
    yield
    print("Shutting down shortfin")
    system.shutdown()


system = System()
app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(value: int, request: Request):
    message = RequestMessage(value, FastAPIResponder(request))
    system.request_writer(message)
    return await message.responder.response


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path to use for installing behind path based proxy.",
    )
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    parser.add_argument(
        "--testing-mock-service",
        action="store_true",
        help="Enable the mock testing service",
    )
    parser.add_argument(
        "--device-uri", type=str, default="local-task", help="Device URI to serve on"
    )

    args = parser.parse_args(argv)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
