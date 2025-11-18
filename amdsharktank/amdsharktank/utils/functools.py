# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable


def compose(*callables: * tuple[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """Compose multiple callables into a single callable that chains them.

    E.g.
    ```
    h = compose(f, g)
    ```
    is equivalent to
    ```
    def h(*args, **kwargs):
        f_res = f(*args, **kwargs)
        if not isinstance(f_res, tuple):
            f_res = (f_res,)
        return g(*f_res)
    ```
    """

    def composed(*args, **kwargs) -> Any:
        res = callables[0](*args, **kwargs)
        for c in callables[1:]:
            if not isinstance(res, tuple):
                res = (res,)
            res = c(*res)
        return res

    return composed
