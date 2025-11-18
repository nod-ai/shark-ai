# amdshark Tuner
`libtuner.py` is the core Python script that provides the fundamental functions
for the tuning loop. It imports `candidate_gen.py` for candidate generation.

To implement the full tuning loop, `libtuner.py` requires a separate Python script
that uses the provided `TuningClient` API from `libtuner.py`.

---

## Prerequisites

### [Optional] Using virtual environments:

```shell
cd amdsharktuner
python -m venv .venv
source .venv/bin/activate
```

### Install python dependencies:

**Development dependencies:**
```shell
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

---

## IREE's Python bindings setup:

### Option 1: Using local IREE Python bindings

#### Configuring python

Uninstall existing packages:

```shell
pip uninstall iree-base-compiler iree-base-runtime
```

#### Build with CMake
```shell
# Configure (include other options as needed)
cmake -G Ninja -B ../iree-build/ \
   -DIREE_BUILD_PYTHON_BINDINGS=ON \
   -DPython3_EXECUTABLE="$(which python3)" \
   .

# Build
cmake --build ../iree-build/
```

> [!IMPORTANT]
> Make sure to enable the ROCM and HIP in your cmake configuration.
> See [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings) for the details.

#### Extend environment variables
```shell
source ../iree-build/.env && export PYTHONPATH
export PATH="$(realpath ../iree-build/tools):$PATH"
```

For more details, refer to the [IREE Python bindings guide](https://iree.dev/building-from-source/getting-started/#python-bindings).

---

### Option 2: Using nightly IREE Python bindings
```shell
pip install --upgrade -r requirements-iree.txt
```

---

## Tuning Algorithm
For a detailed explanation, see the [IREE Tuning Overview](https://iree.dev/reference/tuning/#overview).
1. **Generate candidate specs**
   - Uses the [Z3 solver](https://github.com/Z3Prover/z3/wiki#background) to generate all potential tuning candidate configurations, where `libtuner` applies looser constraints than the IREE compiler.
   - `libtuner` shuffles the Z3 solutions using a default random seed (`42`) to prevent the search from getting trapped in a limited subtree.
   - You can control the randomization seed with `--search-space-shuffle-seed <SEED>`, or use `--enable-random-seed` to apply a non-deterministic shuffle seed.

2. **Compile candidates**

3. **Benchmark**
   - Runs a **baseline benchmark**, executed serially across all specified devices.
   - Runs **candidate benchmarks** in parallel across all devices.
     By default, `libtuner` automatically sets the benchmark timeout for each candidate to the maximum baseline result.
   - Performs a **second baseline run** to check for regressions.
   - Returns the **top-performing candidate indices**.
     If no candidate outperforms the baseline, an empty list (`[]`) is returned.

## Examples

For a concrete example, check the [`model_tuner` directory](./model_tuner/) for a sample tuner implemented with `libtuner`.
The [`dispatch example`](model_tuner/README.md) should be a good starting point for most users.
