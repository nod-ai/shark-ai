FROM rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# apt dependencies
RUN apt-get --fix-broken install -y && apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git wget unzip software-properties-common git \
    build-essential curl cmake ninja-build clang lld vim nano \
    gfortran pkg-config libopenblas-dev libssl-dev openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
RUN source "$PYENV_ROOT/completions/pyenv.bash" && \
    pyenv install 3.11.13 && \
    pyenv global 3.11.13 && \
    pyenv rehash

RUN pip install --upgrade pip setuptools wheel && \
    pip install pybind11 'nanobind<2' numpy==1.* pandas && \
    pip install hip-python hip-python-as-cuda -i https://test.pypi.org/simple

# Rust requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
