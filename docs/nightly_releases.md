# Nightly releases

Nightly releases are uploaded to
https://github.com/nod-ai/shark-ai/releases/tag/dev-wheels.

* The "expanded_assets" version of a release page is compatible with the
  `-f, --find-links <url>` options of `pip install`
  ([docs here](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-f)).
  For the "dev-wheels" release above, that page is:
  <https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels>
* These releases are generated using
  [`.github/workflows/build_package.yml`](../.github/workflows/build_packages.yml)
* That workflow runs the
  [`sharktank/build_tools/build_linux_package.sh`](../sharktank/build_tools/build_linux_package.sh)
  and
[`shortfin/build_tools/build_linux_package.sh`](../shortfin/build_tools/build_linux_package.sh)
  scripts
* Workflow history can be viewed at
  <https://github.com/nod-ai/shark-ai/actions/workflows/build_packages.yml>

## Prerequisites

### Operating system

Currently we only officially support Linux with published packages. Windows and
macOS support is possible, but may need additional setup, code changes, and
source builds.

### Python

You will need a recent version of Python.

* As of Nov 1, 2024, sharktank is compatible with Python 3.11. See
  https://github.com/nod-ai/shark-ai/issues/349 for Python 3.12 support.
* As of Nov 4, 2024, shortfin publishes packages for Python 3.11, 3.12, 3.13,
  and 3.13t

For example, to install Python 3.11 on Ubuntu:

```bash
sudo apt install python3.11 python3.11-dev python3.11-venv

which python3.11
# /usr/bin/python3.11
```

> [!TIP]
> Manage multiple Python versions using `pyenv`
> (<https://github.com/pyenv/pyenv>), or the
> [Python Launcher for Windows](https://docs.python.org/3/using/windows.html#python-launcher-for-windows)
> on Windows.

## Quickstart - sharktank

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.11 -m venv 3.11.venv
source 3.11.venv/bin/activate

# Install 'sharktank' package from nightly releases.
pip install sharktank -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels --pre

# Test the installation.
python -c "from sharktank import ops; print('Sanity check passed')"

# Deactivate the virtual environment when done.
deactivate
```

## Quickstart - sharktuner

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.11 -m venv 3.11.venv
source 3.11.venv/bin/activate

# Install 'sharktuner' package from nightly releases.
pip install sharktuner -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels --pre

# Test the installation.
python -c "import sharktuner; print('Sanity check passed')"

# Deactivate the virtual environment when done.
deactivate
```

## Quickstart - shortfin

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.11 -m venv 3.11.venv
source 3.11.venv/bin/activate

# Install 'shortfin' package from nightly releases, optionally with [apps] extra requirements.
pip install shortfin[apps] -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels --pre

# Test the installation.
python -c "import shortfin as sf; print('Sanity check passed')"

# Deactivate the virtual environment when done.
deactivate
```

## Installing newer versions of dependencies

To install all IREE packages from nightly releases:

```bash
pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
  iree-base-compiler \
  iree-base-runtime \
  iree-turbine
```

To install an editable `iree-turbine` package from the latest source:

```bash
pip install --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"
```

## Switching between stable and nightly channels

The [`shark-ai` package on PyPI](https://pypi.org/project/shark-ai/) is a
meta-package that pins specific stable versions of each package that share
at least their major and minor versions:

```bash
pip install shark-ai==2.9.1

pip freeze
# ...
# iree-base-compiler==2.9.0
# iree-base-runtime==2.9.0
# iree-turbine==2.9.0
# ...
# shark-ai==2.9.1
# shortfin==2.9.1
# ...
```

If you attempt to update any individual package outside of those supported
versions, pip will log an error but continue anyway:

```bash
pip install --upgrade --pre \
  -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels \
  shortfin==3.0.0rc20241118

# Looking in links: https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
# Collecting shortfin==3.0.0rc20241118
#   Downloading https://github.com/nod-ai/shark-ai/releases/download/dev-wheels/shortfin-3.0.0rc20241118-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.5 MB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 24.3 MB/s eta 0:00:00
# Installing collected packages: shortfin
#   Attempting uninstall: shortfin
#     Found existing installation: shortfin 2.9.1
#     Uninstalling shortfin-2.9.1:
#       Successfully uninstalled shortfin-2.9.1
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# shark-ai 2.9.1 requires shortfin==2.9.1, but you have shortfin 3.0.0rc20241118 which is incompatible.
# Successfully installed shortfin-3.0.0rc20241118

pip freeze
# ...
# shark-ai==2.9.1
# shortfin==3.0.0rc20241118
# ...
```

Installing the `shark-ai` package again should get back to aligned versions:

```bash
pip install shark-ai==2.9.1
# ...
# Installing collected packages: shortfin
#   Attempting uninstall: shortfin
#     Found existing installation: shortfin 3.0.0rc20241118
#     Uninstalling shortfin-3.0.0rc20241118:
#       Successfully uninstalled shortfin-3.0.0rc20241118
# Successfully installed shortfin-2.9.1

pip freeze
# ...
# shark-ai==2.9.1
# shortfin==2.9.1
# ...
```

You can also uninstall the `shark-ai` package to bypass the error and take full
control of package versions yourself:

```bash
pip uninstall shark-ai

pip freeze
# ...
# (note: no shark-ai package)
# shortfin==2.9.1
# ...

pip install --upgrade --pre \
  -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels \
  shortfin==3.0.0rc20241118

# Looking in links: https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
# Collecting shortfin==3.0.0rc20241118
#   Using cached https://github.com/nod-ai/shark-ai/releases/download/dev-wheels/shortfin-3.0.0rc20241118-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.5 MB)
# Installing collected packages: shortfin
#   Attempting uninstall: shortfin
#     Found existing installation: shortfin 2.9.1
#     Uninstalling shortfin-2.9.1:
#       Successfully uninstalled shortfin-2.9.1
# Successfully installed shortfin-3.0.0rc20241118

pip freeze
# ...
# (note: no shark-ai package)
# shortfin==3.0.0rc20241118
# ...
```

If you ever get stuck, consider creating a fresh
[virtual environment](https://docs.python.org/3/library/venv.html).
