#!/bin/bash -f


if [[ $1 = "--nightly" ]]; then
    pip install -r pytorch-rocm-requirements.txt
    pip install -r requirements.txt -r requirements-iree-pinned.txt -e sharktank/ -e shortfin/
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre iree-base-compiler iree-base-runtime iree-turbine
    pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
  		iree-base-compiler iree-base-runtime --src deps \
  		-e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"

elif [[ $1 = "--stable" ]]; then
    pip install shark-ai[apps]
    pip install scikit-image
    pip install torch --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0,<2.6.0"

elif [[ $1 = "--source" ]]; then
    pip install -r pytorch-rocm-requirements.txt
    pip install -r requirements.txt -r requirements-iree-pinned.txt -e sharktank/ -e shortfin/
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
   -DCMAKE_CXX_COMPILER=clang++ \
   -DIREE_BUILD_PYTHON_BINDINGS=ON \
   -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON \
   -DIREE_ENABLE_LLD=ON \
   -DPYTHON3_EXECUTABLE=$(which python3) ; cmake --build ../iree-build/
	cd -
else
    echo "setenv.sh --nightly : To install nightly release"
    echo "setenv.sh --stable  : To install stable release"
fi
