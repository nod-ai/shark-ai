# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This package contains custom operation-like functions which operate on a mix
of `torch.Tensor` and `InferenceTensor` type hierarchies. Available ops
are defined in `signatures`. Specific implementations are in `_impl` modules.

There is a simple `_registry` which allows multiple implementations to be
registered against a signature. Registration is done by type signature. Any
matching implementations are processed in reverse (salience, def order) order.
The first one that does not return NotImplemented is used.

In this way, core operations can be defined over a mix of tensor-like types
and layouts.
"""

from . import _registry
from amdsharktank.types.tensors import unbox_tensor
from .signatures import *
from .shape import *
from .utils import *

# Ensure that implementations are registered.
# Note that delegation prefers matching ops defined later, so order here
# can be important.
from . import default_impls
from . import custom_impls
from . import quantized_impls

from . import attention_impls

# Comment this out to completely disable optimized quantized implementations.
from . import qconv_impls
from . import qlinear_impls

from . import cpu_impls

# This import should stay at the bottom for compatibility
from . import sharded_impls
