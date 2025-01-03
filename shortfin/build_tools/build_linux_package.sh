#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_package.sh
# One stop build of shortfin Python packages for Linux. The Linux build is
# complicated because it has to be done via a manylinux docker container.
#
# Usage:
# Build everything (all python versions):
#   sudo ./build_tools/build_linux_package.sh
#
# Build specific Python versions to custom directory, with tracing enabled:
#   OVERRIDE_PYTHON_VERSIONS="cp312-cp312 cp313-cp313" \
#   OUTPUT_DIR="/tmp/wheelhouse" \
#   SHORTFIN_ENABLE_TRACING="ON" \
#   sudo -E ./build_tools/build_linux_package.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp312-cp312 cp313-cp313
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories with docker created, root
# owned builds. Sorry - there is no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -xeu -o errtrace

THIS_DIR="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd "$THIS_DIR"/../../ && pwd)"
SCRIPT_NAME="$(basename $0)"
ARCH="$(uname -m)"

PYTHON_VERSIONS="${OVERRIDE_PYTHON_VERSIONS:-cp311-cp311 cp312-cp312 cp313-cp313}"
OUTPUT_DIR="${OUTPUT_DIR:-${THIS_DIR}/wheelhouse}"
CACHE_DIR="${CACHE_DIR:-}"
SHORTFIN_ENABLE_TRACING="${SHORTFIN_ENABLE_TRACING:-ON}"

if [[ "${ARCH}" == "x86_64" ]]; then
  MANYLINUX_DOCKER_IMAGE="${MANYLINUX_DOCKER_IMAGE:-ghcr.io/nod-ai/manylinux_x86_64@sha256:4acf83343706d1e37252d6001ded3c97a73bc38620580f855b4e65e35ddc5681}"
else
  # TODO: publish a multi-platform manylinux image and include more deps in all platforms (rust, ccache, etc.)
  MANYLINUX_DOCKER_IMAGE="${MANYLINUX_DOCKER_IMAGE:-quay.io/pypa/manylinux_2_28_${ARCH}:latest}"
fi

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${MANYLINUX_DOCKER_IMAGE}"

  # Canonicalize paths.
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  echo "Outputting to ${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  # Setup cache as needed.
  extra_args=""
  if ! [ -z "$CACHE_DIR" ]; then
    echo "Setting up host cache dir ${CACHE_DIR}"
    mkdir -p "${CACHE_DIR}/ccache"
    extra_args="${extra_args} -v ${CACHE_DIR}:${CACHE_DIR} -e CACHE_DIR=${CACHE_DIR}"
  fi

  docker run --rm \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "OVERRIDE_PYTHON_VERSIONS=${PYTHON_VERSIONS}" \
    -e "OUTPUT_DIR=${OUTPUT_DIR}" \
    -e "SHORTFIN_ENABLE_TRACING=${SHORTFIN_ENABLE_TRACING}" \
    ${extra_args} \
    "${MANYLINUX_DOCKER_IMAGE}" \
    -- ${THIS_DIR}/${SCRIPT_NAME}

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${OUTPUT_DIR}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Marking git safe.directory"
  git config --global --add safe.directory '*'

  echo "Using python versions: ${PYTHON_VERSIONS}"
  local orig_path="${PATH}"

  # Configure caching.
  if [ -z "$CACHE_DIR" ]; then
    echo "Cache directory not configured. No caching will take place."
  else
    # TODO: include this in the dockerfile we use so it gets cached
    install_ccache

    # TODO: debug low cache hit rate (~30% hits out of 98% cacheable) on CI
    mkdir -p "${CACHE_DIR}"
    CACHE_DIR="$(cd ${CACHE_DIR} && pwd)"
    echo "Caching build artifacts to ${CACHE_DIR}"
    export CCACHE_DIR="${CACHE_DIR}/ccache"
    export CCACHE_MAXSIZE="2G"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  fi

  # Build phase.
  echo "******************** BUILDING PACKAGE ********************"
  for python_version in ${PYTHON_VERSIONS}; do
    python_dir="/opt/python/${python_version}"
    if ! [ -x "${python_dir}/bin/python" ]; then
      echo "ERROR: Could not find python: ${python_dir} (skipping)"
      continue
    fi
    export PATH="${python_dir}/bin:${orig_path}"
    echo ":::: Python version $(python --version)"

    clean_wheels "shortfin" "${python_version}"
    build_shortfin
    run_audit_wheel "shortfin" "${python_version}"

    if ! [ -z "$CACHE_DIR" ]; then
      echo "ccache stats:"
      ccache --show-stats
    fi
  done
}

function install_ccache() {
  # This gets an old version.
  # yum install -y ccache

  CCACHE_VERSION="4.10.2"

  if [[ "${ARCH}" == "x86_64" ]]; then
    curl --silent --fail --show-error --location \
        "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-${ARCH}.tar.xz" \
        --output ccache.tar.xz

    tar xf ccache.tar.xz
    cp ccache-${CCACHE_VERSION}-linux-${ARCH}/ccache /usr/local/bin
  elif [[ "${ARCH}" == "aarch64" ]]; then
    # Latest version of ccache is not released for arm64, built it
    git clone --depth 1 --branch "v${CCACHE_VERSION}" https://github.com/ccache/ccache.git
    mkdir -p ccache/build && cd "$_"
    cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
    ninja
    cp ccache /usr/bin/
  fi
}

function build_shortfin() {
  # Note: The SHORTFIN_ENABLE_TRACING environment variable should have been
  # forwarded from the host environment into Docker above.
  python -m pip wheel --disable-pip-version-check -v -w "${OUTPUT_DIR}" "${REPO_ROOT}/shortfin"
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  # Force wildcard expansion here
  generic_wheel="$(echo "${OUTPUT_DIR}/${wheel_basename}-"*"-${python_version}-linux_${ARCH}.whl")"
  ls "${generic_wheel}"
  echo ":::: Auditwheel ${generic_wheel}"
  auditwheel repair -w "${OUTPUT_DIR}" "${generic_wheel}"
  rm -v "${generic_wheel}"
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels ${wheel_basename} ${python_version}"
  rm -f -v "${OUTPUT_DIR}/${wheel_basename}-"*"-${python_version}-"*".whl"
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
