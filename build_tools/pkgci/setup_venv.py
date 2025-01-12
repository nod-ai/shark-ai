#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a Python venv with shark-ai packages from a workflow run.

There are several modes in which to use this script:

* Within a workflow triggered by `workflow_call`, an artifact action will
  typically be used to fetch relevant package artifacts. Specify the fetched
  location with `--artifact-path=`:

  ```yml
  - uses: actions/download-artifact@fa0a91b85d4f404e444e00e005971372dc801d16 # v4.1.8
    with:
      name: linux_x86_64_release_packages
      path: ${{ env.PACKAGE_DOWNLOAD_DIR }}
  - name: Setup venv
    run: |
      ./build_tools/pkgci/setup_venv.py ${VENV_DIR} \
      --artifact-path=${PACKAGE_DOWNLOAD_DIR}
  ```

* Within a workflow triggered by `workflow_dispatch`, pass `artifact_run_id` as
  an input that developers must specify when running the workflow:

  ```yml
  on:
    workflow_dispatch:
      inputs:
      artifact_run_id:
        type: string
        default: ""

  ...
    steps:
    - name: Setup venv
      run: |
        ./build_tools/pkgci/setup_venv.py ${VENV_DIR} \
        --fetch-gh-workflow=${{ inputs.artifact_run_id }}
  ```

  (Note that these two modes are often combined to allow for workflow testing)

* Locally, the `--fetch-gh-workflow=WORKFLOW_ID` can be used to download and
  setup the venv from a specific workflow run in one step:


  ```bash
  python3.11 ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-gh-workflow=12056182052
  ```

* Locally, the `--fetch-git-ref=GIT_REF` can be used to download and setup the
  venv from the latest workflow run for a given ref (commit) in one step:

  ```bash
  python3.11 ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-git-ref=main
  ```

You must have the `gh` command line tool installed and authenticated if you
will be fetching artifacts.
"""

from glob import glob
from pathlib import Path
from typing import Optional, Dict, Tuple

import argparse
import functools
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Setup venv")
    parser.add_argument(
        "venv_dir", type=Path, help="Directory in which to create the venv"
    )
    parser.add_argument("--artifact-path", help="Path in which to find/fetch artifacts")
    parser.add_argument(
        "--packages",
        help="Comma-delimited list of packages to install, in order",
        default="shark-ai,shortfin,sharktank",
    )
    parser.add_argument(
        "--install-using-index",
        help="The default mode installs with `--no-index` to be sure that only "
        "our packages are installed. Setting this flag removes that option, "
        "more closely matching the behavior that users will see when they "
        "install published packages.",
        action="store_true",
    )

    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument(
        "--fetch-gh-workflow", help="Fetch artifacts from a GitHub workflow"
    )
    fetch_group.add_argument("--fetch-git-ref", help="Fetch artifacts for a git ref")

    args = parser.parse_args(argv)
    return args


def get_latest_workflow_run_id_for_ref(ref: str) -> int:
    print(f"Normalizing ref: {ref}")
    normalized_ref = (
        subprocess.check_output(["git", "rev-parse", ref], cwd=REPO_ROOT)
        .decode()
        .strip()
    )

    print(f"Fetching artifacts for normalized ref: {normalized_ref}")
    base_path = f"/repos/nod-ai/shark-ai"
    workflow_run_args = [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        f"{base_path}/actions/workflows/pkgci.yml/runs?head_sha={normalized_ref}",
    ]
    print(f"Running command to list workflow runs:\n  {' '.join(workflow_run_args)}")
    workflow_run_output = subprocess.check_output(workflow_run_args)
    workflow_run_json_output = json.loads(workflow_run_output)
    if workflow_run_json_output["total_count"] == 0:
        raise RuntimeError("Workflow did not run at this commit")

    latest_run = workflow_run_json_output["workflow_runs"][-1]
    print(f"Found workflow run: {latest_run['html_url']}")
    return latest_run["id"]


@functools.lru_cache
def list_gh_artifacts(run_id: str) -> Dict[str, str]:
    print(f"Fetching artifacts for workflow run {run_id}")
    base_path = f"/repos/nod-ai/shark-ai"
    output = subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            f"{base_path}/actions/runs/{run_id}/artifacts",
        ]
    )
    data = json.loads(output)
    # Uncomment to debug:
    # print(json.dumps(data, indent=2))
    artifacts = {
        rec["name"]: f"{base_path}/actions/artifacts/{rec['id']}/zip"
        for rec in data["artifacts"]
    }
    print("Found artifacts:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")
    return artifacts


def fetch_gh_artifact(api_path: str, file: Path):
    print(f"Downloading artifact {api_path}")
    contents = subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            api_path,
        ]
    )
    file.write_bytes(contents)


def find_venv_python(venv_path: Path) -> Optional[Path]:
    paths = [venv_path / "bin" / "python", venv_path / "Scripts" / "python.exe"]
    for p in paths:
        if p.exists():
            return p
    return None


def install_with_index(python_exe, wheels):
    # Install each of the built wheels, allowing dependencies and an index.
    # Note that --pre pulls in prerelease versions of dependencies too, like
    # numpy. We could try a solution like https://stackoverflow.com/a/76124424.
    for artifact_path, package_name in wheels:
        cmd = [
            "uv",
            "pip",
            "install",
            "--pre",
            "-f",
            str(artifact_path),
            package_name,
            "--python",
            str(python_exe),
        ]
        print(f"\nRunning command: {' '.join([str(c) for c in cmd])}")
        subprocess.check_call(cmd)


def install_without_index(python_exe, packages, wheels):
    # Install each of the built wheels without deps or consulting an index.
    # This is because we absolutely don't want this falling back to anything
    # but what we said.
    for artifact_path, package_name in wheels:
        cmd = [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "-f",
            str(artifact_path),
            "--force-reinstall",
            package_name,
            "--python",
            str(python_exe),
        ]
        print(f"\nRunning command: {' '.join([str(c) for c in cmd])}")
        subprocess.check_call(cmd)

    # Install requirements for the requested packages.
    # Note that not all of these are included in the package dependencies, but
    # developers usually want the test requirements too.
    requirements_files = []
    if "sharktank" in packages:
        requirements_files.append("sharktank/requirements.txt")
        requirements_files.append("sharktank/requirements-tests.txt")
    if "shortfin" in packages:
        requirements_files.append("shortfin/requirements-tests.txt")

    for requirements_file in requirements_files:
        cmd = [
            "uv",
            "pip",
            "install",
            "-r",
            str(REPO_ROOT / requirements_file),
            "--python",
            str(python_exe),
        ]
        print(f"\nRunning command: {' '.join([str(c) for c in cmd])}")
        subprocess.check_call(cmd)


def find_wheel(args, artifact_prefix: str, package_name: str) -> Tuple[Path, str]:
    artifact_path = Path(args.artifact_path)

    def has_package():
        norm_package_name = package_name.replace("-", "_")
        pattern = str(artifact_path / f"{norm_package_name}-*.whl")
        files = glob(pattern)
        return bool(files)

    if has_package():
        return (artifact_path, package_name)

    if not args.fetch_gh_workflow:
        raise RuntimeError(
            f"Could not find package {package_name} to install from {artifact_path}"
        )

    # Fetch.
    artifact_path.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{artifact_prefix}_dev_packages"
    artifact_file = artifact_path / f"{artifact_name}.zip"
    if not artifact_file.exists():
        print(f"Package {package_name} not found. Fetching from {artifact_name}...")
        artifacts = list_gh_artifacts(args.fetch_gh_workflow)
        if artifact_name not in artifacts:
            raise RuntimeError(
                f"Could not find required artifact {artifact_name} in run {args.fetch_gh_workflow}"
            )
        fetch_gh_artifact(artifacts[artifact_name], artifact_file)
    print(f"Extracting {artifact_file}")
    with zipfile.ZipFile(artifact_file) as zip_ref:
        zip_ref.extractall(artifact_path)

    # Try again.
    if not has_package():
        raise RuntimeError(f"Could not find {package_name} in {artifact_path}")
    return (artifact_path, package_name)


def main(args):
    # Look up the workflow run for a ref.
    if args.fetch_git_ref:
        latest_gh_workflow = get_latest_workflow_run_id_for_ref(args.fetch_git_ref)
        args.fetch_git_ref = ""
        args.fetch_gh_workflow = str(latest_gh_workflow)
        return main(args)

    # Make sure we have an artifact path if fetching.
    if not args.artifact_path and args.fetch_gh_workflow:
        with tempfile.TemporaryDirectory() as td:
            args.artifact_path = td
            return main(args)

    # Parse command-delimited list of packages from args.
    packages = args.packages.split(",")
    print("Installing packages:", packages)

    artifact_prefix = f"{platform.system().lower()}_{platform.machine()}"
    wheels = []
    for package_name in packages:
        wheels.append(find_wheel(args, artifact_prefix, package_name))
    print("Installing wheels:", wheels)

    # Set up venv using 'uv' (https://docs.astral.sh/uv/).
    # We could use 'pip', but 'uv' is much faster at installing packages.
    venv_path = args.venv_dir
    python_exe = find_venv_python(venv_path)

    if not python_exe:
        print(f"Creating venv at {str(venv_path)}")

        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        subprocess.check_call(
            ["uv", "venv", str(venv_path), "--python", sys.executable]
        )
        python_exe = find_venv_python(venv_path)
        if not python_exe:
            raise RuntimeError("Error creating venv")

    # Install the PyTorch CPU wheels first to save multiple minutes and a lot of bandwidth.
    cmd = [
        "uv",
        "pip",
        "install",
        "-r",
        str(REPO_ROOT / "pytorch-cpu-requirements.txt"),
        "--python",
        str(python_exe),
    ]
    print(f"\nRunning command: {' '.join([str(c) for c in cmd])}")
    subprocess.check_call(cmd)

    if args.install_using_index:
        install_with_index(python_exe, wheels)
    else:
        install_without_index(python_exe, packages, wheels)

    # Log which packages are installed.
    print("")
    print(f"Checking packages with 'uv pip freeze':")
    subprocess.check_call(
        [
            "uv",
            "pip",
            "freeze",
            "--python",
            str(python_exe),
        ]
    )

    print("")
    print(f"venv setup using uv, activate with:\n  source {venv_path}/bin/activate")

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
