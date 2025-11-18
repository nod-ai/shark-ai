#!/bin/bash

set -eu
set -o pipefail

SCRIPT_DIR=$(dirname $0)

SRC_DIR="$SCRIPT_DIR/../.."
PYTORCH_ROCM=0
IREE_UNPINNED=0

show_help() {
    echo "Usage:"
    echo "$(basename "$0") [-h] [--torch-version <value>] [--pytorch-rocm] [--iree-unpinned] [--help]"
    echo "Args:"
    echo "--torch-version: Version of PyTorch. If omitted will install the default from pytorch-*-requirements.txt."
    echo "--pytorch-rocm: Install PyTorch for ROCm instead of for CPU."
    echo "--iree-unpinned: Install IREE unpinned(latest)/pinned version."
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        --torch-version) TORCH_VERSION="$2"; shift ;;
        --pytorch-rocm) PYTORCH_ROCM=1 ;;
        --iree-unpinned) IREE_UNPINNED=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python -m pip install --no-compile --upgrade pip

if [[ -v TORCH_VERSION ]]; then
    if (($PYTORCH_ROCM)); then
        ROCM_VERSION=$(python "$SRC_DIR/build_tools/torch_rocm_version_map.py" $TORCH_VERSION)
        pip install --no-compile \
            --index-url https://download.pytorch.org/whl/rocm$ROCM_VERSION \
            torch==$TORCH_VERSION+rocm$ROCM_VERSION
    else
        pip install --no-compile \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==$TORCH_VERSION+cpu
    fi
else
    if (($PYTORCH_ROCM)); then
        pip install --no-compile -r "$SRC_DIR/pytorch-rocm-requirements.txt"
    else
        pip install --no-compile -r "$SRC_DIR/pytorch-cpu-requirements.txt"
    fi
fi

if (($IREE_UNPINNED)); then
    pip install --no-compile --upgrade -r "$SRC_DIR/requirements-iree-unpinned.txt"
else
    pip install --no-compile -r "$SRC_DIR/requirements-iree-pinned.txt"
    pip uninstall -y wave-lang
    pip install -f https://github.com/iree-org/wave/releases/expanded_assets/dev-wheels wave-lang --no-index
fi

pip install --no-compile -r "$SRC_DIR/amdsharktank/requirements-tests.txt"
pip install --no-compile -e "$SRC_DIR/amdsharktank"
