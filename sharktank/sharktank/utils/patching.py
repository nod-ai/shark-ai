# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import torch


class Patch:
    """Patches calls to forward, allowing various forms of interception."""

    def patch_child_modules(self, module: torch.nn.Module):
        """Given a network, wraps the forward() method of children.

        Different types of callbacks can be specified to control wrapping:

        * after_forward: Called with (module_name, module, results) after the
        forward function returns. Used for logging results.
        """

        def _patch(name: str, m: torch.nn.Module):
            orig_forward = m.forward

            def wrapper(*args, **kwargs):
                results = orig_forward(*args, **kwargs)
                self.after_forward(name, m, results)
                return results

            m.forward = wrapper

        for name, m in module.named_modules():
            _patch(name, m)

    def after_forward(self, module_name: str, module: torch.nn.Module, results):
        """Called after every patched forward() function with results."""
        ...


class SaveModuleResultTensorsPatch(Patch):
    """Module patch which saves the results of all modules to a safetensors file.

    Duplicate module invocations are suffixed with "#n" where n is the zero
    based call counter.

    Modules that return multiple results or non tensor results are ignored.

    Users must call finalize() once all tensors have been accumulated.
    """

    def __init__(self):
        self.tensors = {}
        # Map of module_name to last used index for duplicated tensors.
        self.duplicate_tensors = {}

    def after_forward(self, module_name: str, module: torch.nn.Module, results):
        if not isinstance(results, torch.Tensor):
            return

        result_tensor = torch.detach(results).contiguous().to(device="cpu").clone()
        if module_name in self.tensors:
            orig_dup = self.tensors[module_name]
            del self.tensors[module_name]
            self.duplicate_tensors[module_name] = 0
            self.tensors[f"{module_name}#0"] = orig_dup
        if module_name in self.duplicate_tensors:
            index = self.duplicate_tensors[module_name] + 1
            self.duplicate_tensors[module_name] = index
            self.tensors[f"{module_name}#{index}"] = result_tensor
        else:
            self.tensors[module_name] = result_tensor

    def save_file(self, output_path: Path):
        """Saves accumulated tensors to the given file."""
        from safetensors.torch import save_file

        save_file(self.tensors, output_path)
