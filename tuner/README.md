# IREE dispatch auto-tuning scripts
`libtuner.py` is the core Python script that provides the fundamental functions
for the tuning loop. It imports `candidate_gen.py` for candidate generation. To
implement the full tuning loop, `libtuner.py` requires a separate Python script
that uses the provided `TuningClient` API from `libtuner.py`.

## Prerequisites
[Optional] Using virtual environments:
```shell
cd tuner
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```
Install python dependencies:
```shell
pip install -r requirements-tuner.txt
pip install -r requirements-dev.txt
```
IREE's Python bindings setup:

Using the local IREE's Python bindings:
   - Building with CMake
     ```shell
     -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON \
     -DIREE_BUILD_PYTHON_BINDINGS=ON \
     -DPython3_EXECUTABLE="$(which python)"
     ```
      Note: IREE build should be ROCM and HIP supported

   - Set environment
      ```shell
      source ../iree-build/.env && export PYTHONPATH
      export PATH="$(realpath ../iree-build/tools):$PATH"
      ```
  For more information, refer to the [IREE documentation]
  (https://iree.dev/building-from-source/getting-started/#python-bindings).

Using the local IREE's Python bindings:
```shell
pip install -r ../requirements-iree-unpinned.txt
```

## Examples

Check the `examples` directory for sample tuners implemented with `libtuner`.
The [`dispatch example`](https://github.com/nod-ai/shark-ai/tree/main/tuner/examples/simple)
should be a good starting point for most users.
