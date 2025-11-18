# Python Deployment

These scripts assist with building Python packages and pushing them to
[PyPI (the Python Package Index)](https://pypi.org/). See also

* The Python Packaging User Guide: <https://packaging.python.org/en/latest/>

## Overview

See comments in scripts for canonical usage. This page includes additional
notes.

### Package building

These scripts build packages:

* [`/amdshark-ai/build_tools/build_linux_package.sh`](/amdshark-ai/build_tools/build_linux_package.sh)
* [`/amdsharktank/build_tools/build_linux_package.sh`](/amdsharktank/build_tools/build_linux_package.sh)
* [`/shortfin/build_tools/build_linux_package.sh`](/shortfin/build_tools/build_linux_package.sh)
* [`/amdsharktuner/build_tools/build_linux_package.sh`](/amdsharktuner/build_tools/build_linux_package.sh)

### Version management

These scripts handle versioning across packages, including considerations like
major, minor, and patch levels (`X.Y.Z`), as well as suffixes like
`rc20241107`:

* [`compute_common_version.py`](./compute_common_version.py)
* [`compute_local_version.py`](./compute_local_version.py)
* [`promote_whl_from_rc_to_final.py`](./promote_whl_from_rc_to_final.py)
* [`write_requirements.py`](./write_requirements.py)

### PyPI deployment

These scripts handle promoting nightly releases packages to stable and pushing
to PyPI:

* [`promote_whl_from_rc_to_final.py`](./promote_whl_from_rc_to_final.py)
* [`pypi_deploy.sh`](./pypi_deploy.sh)

Both of these scripts expect to have the dependencies from
[`requirements-pypi-deploy.txt`](./requirements-pypi-deploy.txt) installed.
This can be easily managed by using a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ./requirements-pypi-deploy.txt
```
