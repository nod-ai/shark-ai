# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin.support.deps import ShortfinDepNotFoundError

try:
    import transformers
except ModuleNotFoundError as e:
    raise ShortfinDepNotFoundError(__name__, "transformers") from e

try:
    import tokenizers
except ModuleNotFoundError as e:
    raise ShortfinDepNotFoundError(__name__, "tokenizers") from e

try:
    import dataclasses_json
except ModuleNotFoundError as e:
    raise ShortfinDepNotFoundError(__name__, "dataclasses-json") from e
