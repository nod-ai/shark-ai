import os

from argparse import (
    ArgumentParser,
    Namespace,
)

from importlib.resources import files

from pathlib import Path

from . import service


class ServiceCLIArguments(Namespace):
    port: int
    inference_server_origin: str


parser = ArgumentParser(description="Serve the browser application")

parser.add_argument(
    "--port",
    type=int,
    default=5173,
    help="The port on which to serve the browser application",
)

parser.add_argument(
    "--inference-server-origin",
    dest="inference_server_origin",
    type=str,
    default="http://localhost:8000",
    help="The location where the inference application is being served",
)

package_for_browser_application = Path(
    str(
        files("shortfin_apps")
        .joinpath("sd")
        .joinpath("browser_application")
        .joinpath("distributable_package")
    )
)

if not os.path.exists(package_for_browser_application):
    raise FileNotFoundError(
        f"Directory for packaged browser application not found: {package_for_browser_application}"
    )

if __name__ == "__main__":
    parsedArguments = parser.parse_args(
        namespace=ServiceCLIArguments(),
    )

    service.run(
        package=package_for_browser_application,
        host="0.0.0.0",
        port=parsedArguments.port,
        inference_server_origin=parsedArguments.inference_server_origin,
    )
