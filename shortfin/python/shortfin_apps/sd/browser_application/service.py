from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

from pathlib import Path

import uvicorn


def run(
    package: Path,
    host: str,
    port: int,
    inference_server_origin: str,
) -> None:
    service_router = APIRouter()

    @service_router.get("/config")
    async def get_config():  # type: ignore
        return JSONResponse(
            {
                "server": {
                    "origin": inference_server_origin,
                },
            }
        )

    service_application = FastAPI()

    service_application.include_router(service_router)

    service_application.mount(
        "/",
        StaticFiles(
            directory=package,
            html=True,
        ),
        name="static",
    )

    uvicorn.run(
        service_application,
        host=host,
        port=port,
    )
