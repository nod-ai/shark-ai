# BOO Tuner

Example of tuning BOO (Bag of Ops) kernels with MIOpen driver commands.

**MIOpen Command Format**: This tuner accepts convolution parameters in MIOpen
driver format. Example command:

```
convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
```

For detailed explanations of MIOpen command parameters, usage examples, and
environment variables (like `BOO_TUNING_SPEC_PATH` and `BOO_CACHE_ON`), see the
[IREE Turbine BOO README](https://github.com/iree-org/iree-turbine/blob/main/iree/turbine/kernel/boo/README.md).

---

## Prerequisites

Follow instructions in [`/amdsharktuner/README.md`](../README.md).

### Set up PYTHONPATH:

```shell
cd amdsharktuner
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# If using local IREE build.
source ../iree-build/.env && export PYTHONPATH
```

### Install PyTorch for ROCm

([IREE Turbine ROCm requirements](https://github.com/iree-org/iree-turbine/blob/main/pytorch-rocm-requirements.txt)):

```shell
pip install --index-url https://download.pytorch.org/whl/rocm6.4 'torch>=2.9'
```

### Install IREE Turbine

(see the
[IREE Turbine README](https://github.com/iree-org/iree-turbine/blob/main/README.md))
- there are two ways:

**Stable releases:**

```shell
pip install iree-turbine
```

**Nightly releases:**

```shell
pip install --find-links https://iree.dev/pip-release-links.html --upgrade --pre iree-turbine
```

---

## Running the Tuner

### Choose a kernel to tune

This example uses convolution kernels specified in MIOpen driver format.

### Recommended Trial Run

For an initial trial to test the tuning loop, use following command:

```shell
cd amdshark-ai/amdsharktuner
python -m boo_tuner \
  --commands-file boo_tuner/example_configs.txt \
  --output-td-spec tuning_spec.mlir \
  --num-candidates 30 \
  --devices hip://0
```

Alternatively, you can run it directly:

```shell
python boo_tuner/boo_tuner.py \
  --commands-file boo_tuner/example_configs.txt \
  --output-td-spec tuning_spec.mlir \
  --num-candidates 30 \
  --devices hip://0
```

> Example input format for multiple devices: use a comma-separated list, such
> as `--devices=hip://0,hip://1`

> [!TIP]
> Use the `--starter-td-spec` option to pass an existing td spec for the run.

---

## Tuning Algorithm

1. Parse MIOpen commands and generate BOO dispatches
2. Extract dispatch benchmarks with `iree-compile`
3. Generate candidate specs, compile, and benchmark
4. Return top candidates

For details on the tuning algorithm, see
[amdshark Tuner Overview](../README.md#tuning-algorithm).

For BOO-specific information (MIOpen format, environment variables), see
[BOO Documentation](https://github.com/iree-org/iree-turbine/tree/main/iree/turbine/kernel/boo).
