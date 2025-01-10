#!/bin/bash

# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script promotes Python packages from nightly releases to PyPI.
#
# Prerequisites:
#   * You will need to have PyPI credentials set up. See
#     https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives
#   * Install requirements, e.g. in a Python virtual environment (venv):
#     `pip install -r requirements-pypi-deploy.txt`
#   * Install python3.13t and install pip. On Ubuntu:
#     ```bash
#     sudo add-apt-repository ppa:deadsnakes
#     sudo apt-get update
#     sudo apt-get install python3.13-nogil
#     python3.13t -m ensurepip --upgrade
#     ```
#   * Choose a release candidate to promote from
#     https://github.com/nod-ai/shark-ai/releases/tag/dev-wheels
#
# Usage:
#   ./pypi_deploy.sh 2.9.0rc20241108

set -euo pipefail

RELEASE="$1"

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
REPO_ROOT="$(cd "$SCRIPT_DIR"/../../ && pwd)"
TMPDIR="$(mktemp --directory --tmpdir shark_platform_pypi_wheels.XXXXX)"
ASSETS_PAGE="https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels"

# TODO: rewrite in Python?

function download_wheels() {
  echo ""
  echo "Downloading wheels for '${RELEASE}'..."

  # sharktank
  python -m pip download sharktank==${RELEASE} \
    --no-deps --python-version 3.11 -f ${ASSETS_PAGE}

  # shortfin
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.11 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.12 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.13 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.13 -f ${ASSETS_PAGE}
  # TODO: fetch 3.13t using the same `python` somehow
  #   * https://pip.pypa.io/en/stable/cli/pip_download/
  #   * https://py-free-threading.github.io/installing_cpython/
  #   * https://pip.pypa.io/en/stable/installation/
  python3.13t -m pip download shortfin==${RELEASE} --no-deps -f ${ASSETS_PAGE}

  # TODO: shark-ai meta package when it is published to nightlies

  echo ""
  echo "Downloaded wheels:"
  ls
}

function edit_release_versions() {
  echo ""
  echo "Editing release versions..."
  for file in *
  do
    ${SCRIPT_DIR}/promote_whl_from_rc_to_final.py ${file} --delete-old-wheel
  done

  echo "Edited wheels:"
  ls
}

function upload_wheels() {
  # TODO: list packages that would be uploaded, pause, prompt to continue
  echo ""
  echo "Uploading wheels:"
  ls
  twine upload --verbose *
}

function build_shark_ai_meta_package() {
  # TODO: download meta package from nightly releases instead of this
  #   Be aware that nightly releases pin other dependencies via the
  #   generated `requirements.txt` compared to stable releases.
  echo ""

  # TODO: rework `write_requirements.py` to use the versions from the downloaded whls?
  echo "Computing local versions for sharktank and shortfin..."
  ${SCRIPT_DIR}/compute_local_version.py ${REPO_ROOT}/sharktank -stable --write-json
  ${SCRIPT_DIR}/compute_local_version.py ${REPO_ROOT}/shortfin -stable --write-json

  echo "Computing common version for shark-ai meta package..."
  ${SCRIPT_DIR}/compute_common_version.py --stable-release --write-json

  echo "Writing requirements for shark-ai meta package..."
  ${SCRIPT_DIR}/write_requirements.py

  echo "Building shark-ai meta package..."
  ${REPO_ROOT}/shark-ai/build_tools/build_linux_package.sh

  # TODO: This is error-prone. We only want to publish the whl for this release.
  #   Copy instead? Specify exact file name? Clear directory before building?
  mv ${REPO_ROOT}/shark-ai/build_tools/wheelhouse/* .
}

function main() {
  echo "Changing into ${TMPDIR}"
  cd "${TMPDIR}"
  # TODO: check_requirements (using pip)

  download_wheels
  edit_release_versions
  build_shark_ai_meta_package
  upload_wheels
}

main
