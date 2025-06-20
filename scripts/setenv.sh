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
else
    echo "setenv.sh --nightly : To install nightly release"
    echo "setenv.sh --stable  : To install stable release"
fi
