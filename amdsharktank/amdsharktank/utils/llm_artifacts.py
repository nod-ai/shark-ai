# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from iree.compiler import compile_str
from iree.turbine.aot import ParameterArchiveBuilder
from amdsharktank.examples.export_paged_llm_v1 import export_llm_v1
from amdsharktank.layers.configs import LlamaModelConfig
from amdsharktank.models.llm.export import ExportConfig
from amdsharktank.types.theta import Theta
from amdsharktank.utils.llm_utils import IreeInstance


class LlmArtifactBuilder:
    def __init__(self, theta: Theta, llama_config: LlamaModelConfig):
        self._theta = theta
        self._llama_config = llama_config
        self._ir = None
        self._server_config = None
        self._vmfb = None

    def export(self, export_config: ExportConfig):
        output, server_config = export_llm_v1(
            theta=self._theta,
            llama_config=self._llama_config,
            export_config=export_config,
        )
        self._ir = output.mlir_module.get_asm()
        self._server_config = server_config

    def compile(self, args):
        if self._ir is None:
            raise Exception("Must export before compiling")
        self._vmfb = compile_str(self._ir, extra_args=args)

    def instance(self, devices):
        if self._vmfb is None:
            raise Exception("Must compile before invoke")

        builder = ParameterArchiveBuilder()
        properties = self._llama_config.hp.to_gguf_props()
        self._theta.add_tensors_to_archive(
            irpa=builder, inference_tensor_metas=properties
        )
        index = builder.index

        return IreeInstance(
            devices=devices,
            vmfb=self._vmfb,
            parameters=index,
            config=self._llama_config,
        )
