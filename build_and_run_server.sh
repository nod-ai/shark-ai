pip uninstall -y iree-base-compiler iree-base-runtime


cmake -G Ninja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_HAL_DRIVER_HIP=ON \
    -DIREE_TARGET_BACKEND_ROCM=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DIREE_TARGET_BACKEND_CUDA=OFF

cmake --build ../iree-build/

export SHORTFIN_DEV_MODE=ON
export SHORTFIN_IREE_SOURCE_DIR="$HOME/iree"
export SHORTFIN_APPS_LOG_LEVEL=DEBUG
export PATH="$HOME"/iree-build/tools:$PATH
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048

source $HOME/iree-build/.env && export PYTHONPATH

pushd shortfin && python ./dev_me.py --iree $HOME/iree --build-type=RelWithDebInfo --no-tracing
popd

# prep artifacts

export SHORTFIN_DEV_MODE=ON
export SHORTFIN_IREE_SOURCE_DIR="$HOME/iree"
export SHORTFIN_APPS_LOG_LEVEL=DEBUG
export PATH="$HOME"/iree-build/tools:$PATH
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048

source $HOME/iree-build/.env && export PYTHONPATH
pyenv activate shark-virtualenv
export EXPORT_DIR=~/performant_llama31_8b
mkdir -p $EXPORT_DIR
export MODEL_PARAMS_PATH=/home/xidaren2/stephen_artifacts_bs4/meta-llama-3.1-8b-instruct.f16.gguf
export TOKENIZER_PATH=/home/xidaren2/stephen_artifacts_bs4/tokenizer.json
export MLIR_PATH=$EXPORT_DIR/model.mlir
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json
export VMFB_PATH=$EXPORT_DIR/model.vmfb
export BS_PREFILL=4
export BS_DECODE=32
cp $TOKENIZER_PATH $EXPORT_DIR/
cp /home/xidaren2/stephen_artifacts_bs4/tokenizer_config.json $EXPORT_DIR/
cp /home/xidaren2/stephen_artifacts_bs4/config.json $EXPORT_DIR/
echo "Exporting model to MLIR..."
python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs-prefill=$BS_PREFILL \
  --bs-decode=$BS_DECODE
echo "Compiling model to VMFB..."
echo "vvv Compiling using vvv"
which iree-compile
echo "^^^ Compiling using ^^^"
iree-compile $MLIR_PATH \
 --iree-hal-target-backends=rocm \
 --iree-hip-target=gfx942 \
 -o $VMFB_PATH

# Wait for user confirmation
echo "Model artifacts prepared in $EXPORT_DIR"
echo "Press Enter to run server or Ctrl+C to cancel..."

SHORTFIN_APPS_LOG_LEVEL=INFO SHORTFIN_LLM_BATCHER_TYPE=instant python -m shortfin_apps.llm.server --tokenizer_json=/home/xidaren2/performant_llama31_8b/tokenizer.json --model_config=/home/xidaren2/performant_llama31_8b/config.json --vmfb=/home/xidaren2/performant_llama31_8b/model.vmfb --parameters=/home/xidaren2/stephen_artifacts_bs4/meta-llama-3.1-8b-instruct.f16.gguf --device=hip --n_beams 8 --device_ids 0 --port=32567 |& tee beam.log
