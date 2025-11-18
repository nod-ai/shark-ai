# amdshark Developer Guide

Each sub-project has its own developer guide. If you would like to work across
projects, these instructions should help you get started:


### Install Dependencies

Install shortfin dependencies
```bash
sudo apt update && sudo apt install -y clang lld
```

### Prepare your python environment

Install:

```bash
sudo apt install python-is-python3 python3.11-venv python3.11-dev
```

<details>

<summary> Or, alternatively, use `pyenv` to manage a separate python installation for more control over its version: </summary>


The following instructions are taken from pyenv's guide here: https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv

First, install pyenv and its dependencies.

```bash
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
```

Then, make pyenv available by adding the below to your `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

Finally, install a pyenv-managed version of python

```bash
pyenv install 3.12 # or whichever python version you'd like
pyenv local 3.12
```

Now, your python, pip, and venv should be managed by pyenv instead.

</details>

### Setup a venv

We recommend setting up a Python
[virtual environment (venv)](https://docs.python.org/3/library/venv.html).
The project is configured to ignore `.venv` directories, and editors like
VSCode pick them up by default.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install PyTorch for your system

If no explicit action is taken, the default PyTorch version will be installed.
This will give you a current CUDA-based version, which takes longer to download
and includes other dependencies that amdshark does not require. In order to run eager mode inference in amdsharktank on an AMD GPU, `torch+rocm` installation is required.

Follow these commands to install the appropriate torch version:

* *CPU:*

  ```bash
  pip install -r pytorch-cpu-requirements.txt
  ```

* *ROCM:*

  ```bash
  pip install -r pytorch-rocm-requirements.txt
  ```

* *Other:* see instructions at <https://pytorch.org/get-started/locally/>.

### Install development packages

```bash
# Install editable local projects.
pip install -r requirements.txt -e amdsharktank/ -e shortfin/

# Install the latest nightly release of iree-turbine, alond with
# nightly versions of iree-base-compiler and iree-base-runtime.
pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
  iree-base-compiler iree-base-runtime iree-turbine
```

You can also install an editable iree-turbine dep:
```bash
# Optionally clone and install the latest editable iree-turbine dep in deps/.
pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
  iree-base-compiler iree-base-runtime --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"
```

See also: [nightly_releases.md](nightly_releases.md).

### Running tests

```bash
pip install -r shortfin/requirements-tests.txt
pytest amdsharktank
pytest shortfin
pytest app_tests/integration_tests
```

### Optional: pre-commits and developer settings

This project is set up to use the `pre-commit` tooling. To install it in
your local repo, run: `pre-commit install`. After this point, when making
commits locally, hooks will run. See https://pre-commit.com/
