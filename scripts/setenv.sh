#!/bin/bash -f
set -e

export BUILD_TYPE="stable"
export IREE_COMMIT_HASH="main"
export IREE_REMOTE_REPO="iree-org/iree"
export amdshark_AI_REMOTE_REPO="nod-ai/amdshark-ai"
export amdshark_AI_COMMIT_HASH="main"
SCRIPT_DIR=$(dirname $(realpath "$0"))
amdshark_AI_ROOT_DIR=${SCRIPT_DIR}/../

while [[ "$1" != "" ]]; do
    case "$1" in
        --stable)
            export BUILD_TYPE="stable"
            ;;
        --nightly)
            export BUILD_TYPE="nightly"
            ;;
        --nightly-cpu)
            export BUILD_TYPE="nightly-cpu"
            ;;
        --tom)
            export BUILD_TYPE="tom"
            ;;
        --source)
            export BUILD_TYPE="source"
            ;;
        --iree-commit-hash)
            shift
            export IREE_COMMIT_HASH=$1
            ;;
        --iree-remote-repo)
            shift
            export IREE_REMOTE_REPO=$1
            ;;
        --amdshark-ai-commit-hash)
            shift
            export amdshark_AI_COMMIT_HASH=$1
            ;;
        --amdshark-ai-remote-repo)
            shift
            export amdshark_AI_REMOTE_REPO=$1
            ;;
        -h|--help)
            echo "Usage: $0 [--<different flags>] "
            echo "setenv.sh --nightly : To install nightly release"
            echo "setenv.sh --stable  : To install stable release"
            echo "setenv.sh --tom  : To install with TOM IREE and amdshark-ai"
            echo "setenv.sh --source  : To install from IREE and amdshark-ai source"
            echo "setenv.sh --nightly-cpu : To install nightly release with pytorch for cpu"
            echo "--iree-commit-hash <hash> : To install IREE with specified commit"
            echo "--iree-remote-repo <org/repo> To install with specified IREE fork. Defaults to iree-org/iree"
            echo "--amdshark-ai-commit-hash <hash> : To install amdshark-ai with specified commit"
            echo "--amdshark-ai-remote-repo <org/repo> : To install with specified amdshark-ai fork. Defaults to nod-ai/amdshark-ai"
            exit 0
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

mkdir -p ${SCRIPT_DIR}/../output_artifacts

if [[ $BUILD_TYPE = "nightly" ]]; then
    pip install -r pytorch-rocm-requirements.txt
    pip install amdsharktank -f https://github.com/nod-ai/amdshark-ai/releases/expanded_assets/dev-wheels --pre
    pip install shortfin[apps] -f https://github.com/nod-ai/amdshark-ai/releases/expanded_assets/dev-wheels --pre
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre iree-base-compiler iree-base-runtime iree-turbine
    pip uninstall --y wave-lang
    pip install -f https://github.com/iree-org/wave/releases/expanded_assets/dev-wheels wave-lang --no-index
    pip install scikit-image

elif [[ $BUILD_TYPE = "stable" ]]; then
    pip install amdshark-ai[apps]
    pip install scikit-image
    pip install torch --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0,<2.6.0"

elif [[ $BUILD_TYPE = "nightly-cpu" ]]; then
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    pip install amdsharktank -f https://github.com/nod-ai/amdshark-ai/releases/expanded_assets/dev-wheels --pre
    pip install shortfin[apps] -f https://github.com/nod-ai/amdshark-ai/releases/expanded_assets/dev-wheels --pre
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre iree-base-compiler iree-base-runtime iree-turbine
    pip uninstall --y wave-lang
    pip install -f https://github.com/iree-org/wave/releases/expanded_assets/dev-wheels wave-lang --no-index
    pip install scikit-image

elif [[ $BUILD_TYPE = "source" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

    # Create and install amdsharktank and shortfin wheels
    if [ ! -d "amdshark_ai_source" ]; then
        git clone https://github.com/nod-ai/amdshark-ai.git amdshark_ai_source
    fi
    cd amdshark_ai_source
    if git config remote.fork_user.url > /dev/null; then
        git remote remove fork_user
    fi
    git remote add fork_user https://github.com/${amdshark_AI_REMOTE_REPO}
    git fetch fork_user
    git checkout ${amdshark_AI_COMMIT_HASH}

    pip install -r requirements.txt
    # Install amdsharktank and shortfin
    pip install -v amdsharktank/ shortfin/

    ## Install wave
    rm -rf wave
    git clone https://github.com/iree-org/wave.git
    cd wave
    pip install -r requirements.txt -e .
    echo -n "Wave : " >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    git log -1 --pretty=%H >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    cd $amdshark_AI_ROOT_DIR

    python -c "from amdsharktank import ops; print('amdsharktank sanity check passed')"
    if [[ $? != 0 ]]; then
        echo "Failed to install amdsharktank wheel"
        exit 1
    fi
    python -c "import shortfin as sf; print('Shortfin sanity check passed')"
    if [[ $? != 0 ]]; then
        echo "Failed to install shortfin wheel"
        exit 1
    fi
    echo -n "amdshark-AI (${amdshark_AI_REMOTE_REPO}) : " >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    git log -1 --pretty=%H >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    cd $amdshark_AI_ROOT_DIR

    ## Create and install IREE compiler and runtime wheels
    rm -rf iree
    git clone https://github.com/iree-org/iree.git && cd iree
    git remote add fork_user https://github.com/${IREE_REMOTE_REPO}
    git fetch fork_user
    git checkout ${IREE_COMMIT_HASH}
    git submodule update --init
    export IREE_HAL_DRIVER_HIP=ON
    export IREE_TARGET_BACKEND_ROCM=ON
    pip install -v compiler/ runtime/
    echo -n "IREE (${IREE_REMOTE_REPO}) :" >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    git log -1 --pretty=%H >> ${SCRIPT_DIR}/../output_artifacts/version.txt
    cd $amdshark_AI_ROOT_DIR
    rm -rf iree

    ## Install editable local iree turbine
    rm -rf iree-turbine
    git clone https://github.com/iree-org/iree-turbine.git
    cd iree-turbine
    pip install -e .
    cd $amdshark_AI_ROOT_DIR

elif [[ $BUILD_TYPE = "tom" ]]; then
    pip install -r pytorch-rocm-requirements.txt
    pip install -r requirements.txt -r requirements-iree-pinned.txt -e amdsharktank/ -e shortfin/
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre iree-base-compiler iree-base-runtime iree-turbine
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
          iree-base-compiler iree-base-runtime --src deps \
          -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"
    pip uninstall -y iree-base-compiler iree-base-runtime
    git clone https://github.com/iree-org/iree.git
    cd iree
        git submodule update --init
        cmake -G Ninja -B ../iree-build/ -S . \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DIREE_ENABLE_ASSERTIONS=ON \
       -DIREE_ENABLE_SPLIT_DWARF=ON \
       -DIREE_ENABLE_THIN_ARCHIVES=ON \
       -DCMAKE_C_COMPILER=clang \
       -DIREE_HIP_TEST_TARGET_CHIP= \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DIREE_BUILD_PYTHON_BINDINGS=ON \
       -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON \
       -DIREE_ENABLE_LLD=ON \
       -DPYTHON3_EXECUTABLE=$(which python3) ; cmake --build ../iree-build/
    cd -
else
    echo "Invalid build type specified"
    exit 1
fi
