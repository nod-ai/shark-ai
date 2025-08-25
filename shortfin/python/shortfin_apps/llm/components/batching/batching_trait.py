# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from ..config_struct import ModelParams


# BatchingTrait is an interface that ensures that the batching
# engine always has the methods needed by the facade class.
# This is used second in inheritance, usually with another base
# class. Since this is not a concrete class and Python implements
# a neat MRO, we can safely use multiple inheritance without worrying
# about complications like the diamond problem.
class BatchingTrait(ABC):
    @abstractmethod
    def shutdown(self, *args, **kwargs):
        pass

    @abstractmethod
    def launch(self, *args, **kwargs):
        pass

    @abstractmethod
    def submit(self, *args, **kwargs):
        pass

    @abstractmethod
    def reserve_workload(self, *args, **kwargs):
        pass

    @staticmethod
    def create(*args, **kwargs) -> "BatchingTrait":
        pass

    @abstractmethod
    def get_model_params(self) -> ModelParams:
        pass
