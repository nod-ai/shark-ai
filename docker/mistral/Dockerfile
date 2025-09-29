FROM rocm/dev-ubuntu-22.04:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/shark-ai/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=300

ARG SHARK_AI_COMMIT

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    git \
    wget \
    unzip \
    software-properties-common \
    build-essential \
    curl \
    cmake \
    ninja-build \
    clang \
    lld \
    vim \
    nano \
    gfortran \
    pkg-config \
    aria2 \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Install shark-ai and its dependencies
RUN git clone https://github.com/nod-ai/shark-ai.git && \
    cd shark-ai && \
    git checkout ${SHARK_AI_COMMIT} && \
    uv venv -p python3.12 && . .venv/bin/activate && \
    uv pip install pip setuptools wheel pybind11 'nanobind<2' numpy~=1.10 pandas && \
    uv pip install  hip-python hip-python-as-cuda -i https://test.pypi.org/simple && \
    uv pip install -r pytorch-rocm-requirements.txt && \
    uv pip install -r requirements.txt -r requirements-iree-pinned.txt && \
    uv pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
    iree-base-compiler \
    iree-base-runtime \
    iree-turbine && \
    uv pip install  -e sharktank/ -e shortfin/

WORKDIR /shark-ai/shortfin

COPY ./docker/mistral/ .

# # ENV ROCR_VISIBLE_DEVICES=0,1
ENV SOURCE_DIR=/weights/quark
ENV CONFIG_DIR=/config/quark

CMD . /shark-ai/.venv/bin/activate && bash download_weights.sh \
    && bash build_and_benchmark.sh \
    && bash run_server.sh
