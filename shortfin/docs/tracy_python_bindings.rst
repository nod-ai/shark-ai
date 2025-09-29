.. Copyright 2024 Advanced Micro Devices, Inc.

.. _tracy_python_bindings:

Tracy Python Bindings in Shortfin
===============================

This document explains how to build and import Tracy Python bindings for profiling Shortfin applications.

Building Tracy Python Bindings
-----------------------------

Prerequisites
~~~~~~~~~~~~~

- Python version: The bindings will be built for the Python version used during the build process. Make sure to use the same Python version for building and importing, or you might encounter ``ModuleNotFoundError: No module named 'tracy_client.TracyClientBindings'``.

Build Steps
~~~~~~~~~~

1. Build Shortfin with Tracy Python bindings enabled::

    python dev_me.py --iree=/path/to/iree --tracy-python-bindings

2. Verify the build completed successfully. The Tracy Python bindings should be built and copied to the following location::

    /path/to/shortfin/build/cmake/tracy/python/tracy_client/

Importing Tracy Python Bindings
-----------------------------

Location of Bindings
~~~~~~~~~~~~~~~~~~

After a successful build, the Tracy Python bindings are located at::

    /path/to/shortfin/build/cmake/tracy/python/tracy_client/

This directory contains:

- TracyClientBindings.cpython-XXX.so (native extension module)
- Python wrapper files (tracy.py, scoped.py, etc.)

Setting Up PYTHONPATH
~~~~~~~~~~~~~~~~~~~

To import the Tracy Python bindings, you need to add their location to your PYTHONPATH::

    import os
    import sys

    # Add Tracy Python bindings to PYTHONPATH
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:/path/to/shortfin/build/cmake/tracy/python"

    # Now you can import tracy_client
    import tracy_client

Replace ``/path/to/shortfin`` with the actual path to your Shortfin repository.

Common Import Issues
~~~~~~~~~~~~~~~~~~

Module Not Found Error
^^^^^^^^^^^^^^^^^^^^^^

If you encounter an error like::

    ModuleNotFoundError: No module named 'tracy_client.TracyClientBindings'

This indicates that Python can't find the native extension module within the package structure. Check that:

1. Your PYTHONPATH is set correctly to include the parent directory of tracy_client (not the tracy_client directory itself)
2. The TracyClientBindings.cpython-XXX.so file exists in the tracy_client directory
3. You're using the same Python version that was used during the build process

Python Version Mismatch
^^^^^^^^^^^^^^^^^^^^^

The Tracy Python bindings must be used with the same Python version they were built with. If you encounter an error like::

    ImportError: Python version mismatch: module was compiled for Python 3.12, but the interpreter version is incompatible: 3.10.12

You need to:

1. Use the same Python version that was used during the build, or
2. Rebuild the Tracy Python bindings with your current Python version

More on Tracy
---------

For information on how to use the Tracy Python bindings, refer to:

- `Tracy's One Big PDF of Documentation <https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf>`_
- `Tracy Python Bindings (from the Tracy repo) <https://github.com/wolfpld/tracy/tree/master/python>`_
