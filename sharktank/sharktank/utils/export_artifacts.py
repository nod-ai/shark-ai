# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from amdsharktank.utils.iree import get_iree_compiler_flags_from_object
from amdsharktank.utils.evaluate import *
from amdsharktank.utils.logging import get_logger

if TYPE_CHECKING:
    from amdsharktank.layers import LlamaModelConfig

logger = get_logger(__name__)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)


class ExportArtifactsException(Exception):
    """Base exception for export artifacts errors that preserves the command line and error output."""

    def __init__(
        self, process: subprocess.CompletedProcess, cwd: str, export_stage: str
    ):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)
        super().__init__(
            f"Error invoking {export_stage}\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n"
            f"Stdout diagnostics:\n{outs}\n"
            f"Invoked with:\n"
            f"\tcd {cwd} && {process.args}\n\n"
        )


class ExportMlirException(ExportArtifactsException):
    """amdshark-ai export MLIR exception."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        super().__init__(process, cwd, export_stage="export_paged_llama_v1.py")


class IreeBenchmarkException(ExportArtifactsException):
    """IREE benchmark runtime exception."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        super().__init__(process, cwd, export_stage="iree-benchmark-module")


class IreeCompileException(ExportArtifactsException):
    """IREE compiler exception."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        super().__init__(process, cwd, export_stage="iree-compile")


class IreeRunException(ExportArtifactsException):
    """Runtime exception."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        super().__init__(process, cwd, export_stage="iree-run-module")


class IrpaShardException(ExportArtifactsException):
    """IRPA sharding exception."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        super().__init__(process, cwd, export_stage="shard_llm_dataset.py")


class ExportArtifacts:
    """
    Class to manage export artifacts for LLM models, including sharding IRPA files,
    exporting to MLIR, compiling to VMFB, and running the resulting VMFB.
    """

    def __init__(
        self,
        *,
        irpa_path: str | Path,
        attention_kernel: str | None = None,
        matmul_kernel: str | None = None,
        tensor_parallelism_size: int,
        pipeline_parallelism_size: int,
        block_seq_stride: int,
        iree_hal_target_device: str,
        iree_hip_target: str | None = None,
        iree_hal_local_target_device_backends: str | None = None,
        interleave_rotary: bool = True,
        activation_dtype: str = "float16",
        attention_dtype: str = "float16",
        kv_cache_dtype: Optional[str | Path] = None,
        output_mlir: Optional[str | Path] = None,
        output_config: Optional[str | Path] = None,
        output_vmfb: Optional[str | Path] = None,
        output_name: Optional[str | Path] = None,
        cwd: Optional[str | Path] = None,
        hip_device_id: str,
        use_qk_norm: bool = False,
        attention_chunk_size: int | None = None,
    ):
        self.tmp_dir = Path(tempfile.mkdtemp(type(self).__qualname__))
        self.cwd = Path(cwd if cwd is not None else self.tmp_dir)
        self.cwd.mkdir(parents=True, exist_ok=True)

        self.irpa_path = Path(irpa_path).resolve()
        # Note: The following 3 paramaters are used by `get_iree_compiler_flags_from_object`
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.iree_hal_local_target_device_backends = (
            iree_hal_local_target_device_backends
        )
        self.attention_kernel = attention_kernel
        self.matmul_kernel = matmul_kernel
        self.tensor_parallelism_size = tensor_parallelism_size
        self.pipeline_parallelism_size = pipeline_parallelism_size
        self.parallelism_size = (
            self.tensor_parallelism_size * self.pipeline_parallelism_size
        )
        self.block_seq_stride = block_seq_stride
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.interleave_rotary = interleave_rotary
        self.hip_device_id = hip_device_id
        self.use_qk_norm = use_qk_norm
        self.attention_chunk_size = attention_chunk_size

        self.output_mlir = output_mlir
        self.output_config = output_config
        self.output_vmfb = output_vmfb

        if output_name is not None:
            self.output_name = Path(output_name)
        else:
            self.output_name = self.cwd / (
                str(self.irpa_path).split("/")[-1].rsplit(".", 1)[0].replace(".", "_")
                + (f"_{self.attention_kernel}" if self.attention_kernel else "")
                + (
                    f"_pp{self.pipeline_parallelism_size}"
                    if self.pipeline_parallelism_size > 1
                    else ""
                )
            )

    def __del__(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def from_config(
        config: "LlamaModelConfig", /, **init_kwargs: dict[str, Any]
    ) -> "ExportArtifacts":
        """
        Creates an ExportArtifacts instance from a LlamaModelConfig object.

        Args:
            config: The LlamaModelConfig object containing model configuration.
            init_kwargs: Additional keyword arguments to pass to the ExportArtifacts constructor.
        """
        properties = config.to_properties()
        kv_cache_dtype = (
            properties["kv_cache_dtype"] if "kv_cache_dtype" in properties else None
        )
        return ExportArtifacts(
            attention_kernel=config.attention_kernel,
            matmul_kernel=config.matmul_kernel,
            tensor_parallelism_size=config.tensor_parallelism_size,
            pipeline_parallelism_size=config.pipeline_parallelism_size,
            block_seq_stride=config.block_seq_stride,
            interleave_rotary=config.hp.rope_interleave_emb,
            activation_dtype=properties["activation_dtype"],
            attention_dtype=properties["attention_dtype"],
            kv_cache_dtype=kv_cache_dtype,
            **init_kwargs,
        )

    def _prepare_params_and_devices(self) -> tuple[List[str], List[str]]:
        """
        Prepares the parameters and devices for the IREE commands based on the IRPA path and HIP device ID.
        Automatically handles tensor parallelism and multiple devices.

        Requires at least as many devices on the systems as the parallelism size.

        Returns:
            A tuple containing:
            - A list of parameters for the IREE command.
            - A list of device arguments for the IREE command.
            - A list of environment variables for the ROCr visible devices.
        """
        if self.parallelism_size > 1:
            base_irpa_path, _ = os.path.splitext(self.irpa_path)
            rocr_visible_devices = [
                f"ROCR_VISIBLE_DEVICES={','.join(str(i) for i in range(self.parallelism_size))}"
            ]
            params = [f"--parameters=model={base_irpa_path}.irpa"]
            if self.tensor_parallelism_size > 1:
                params += [
                    f"--parameters=model={base_irpa_path}.rank{i}.irpa"
                    for i in range(self.tensor_parallelism_size)
                ]
            devices = [f"--device=hip://{i}" for i in range(self.parallelism_size)]
        else:
            hip_device_arg = int(self.hip_device_id.split("://")[1])
            rocr_visible_devices = [
                f"ROCR_VISIBLE_DEVICES={','.join(str(i) for i in range(hip_device_arg + 1))}"
            ]
            params = [f"--parameters=model={self.irpa_path}"]
            devices = [f"--device={self.hip_device_id}"]

        return params, devices, rocr_visible_devices

    def _run_cmd(
        self,
        cmd: str,
        run_msg: str,
        success_msg: str,
        exception: ExportArtifactsException,
    ) -> None:
        """
        Helper function to run a command and handle exceptions.

        Args:
            cmd: The command to run as a string.
            run_msg: Message to log before running the command.
            success_msg: Message to log if the command runs successfully.
            exception: The exception class to raise if the command fails.
        """
        logger.info(f"{run_msg}:\n" f"cd {self.cwd} && {cmd}")

        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=self.cwd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise exception(proc, self.cwd)
        else:
            logger.info(f"{success_msg}:\n" f"{proc.stdout}")

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time_ns()
            result = func(*args, **kwargs)
            end = time.time_ns()
            time_taken = calc_time(start, end)
            func_name = func.__name__
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

    @timeit
    def shard_irpa_file(self) -> None:
        """
        Shards the IRPA file into multiple smaller files based on the tensor parallelism size.
        Replaces the orignal IRPA file path with that of the sharded version.

        Raises an IrpaShardException if the sharding fails for some reason.
        """
        irpa_path = self.irpa_path
        output_irpa = irpa_path.with_name(f"{irpa_path.stem}_sharded{irpa_path.suffix}")
        self.irpa_path = output_irpa

        shard_irpa_args = [
            "python3",
            "-m",
            "amdsharktank.examples.sharding.shard_llm_dataset",
            "--irpa-file",
            irpa_path,
            "--output-irpa-file",
            output_irpa,
            "--tensor-parallelism-size",
            str(self.tensor_parallelism_size),
        ]

        self._run_cmd(
            cmd=subprocess.list2cmdline(shard_irpa_args),
            run_msg="Sharding irpa file",
            success_msg="Sharded irpa file successfully",
            exception=IrpaShardException,
        )

    @timeit
    def export_llm_to_mlir(
        self,
        *,
        batch_size: int,
        skip_decode: bool = False,
    ) -> None:
        """
        Exports the LLM to MLIR format using the `export_paged_llm_v1` script.

        Raises an ExportMlirException if the export fails for some reason.

        Args:
            batch_size: The batch size to use for prefill and decode.
            skip_decode: If True, skips the decoding step during export.
        """

        if self.output_mlir is not None and self.output_config is not None:
            logger.info(f" Using pre-exported mlir: {self.output_mlir}")
            logger.info(f" Using pre-exported config json: {self.output_config}")
            return
        else:
            self.output_mlir = self.output_name.with_suffix(".mlir")
            self.output_config = self.output_name.with_suffix(".json")

        export_args = [
            "python3",
            "-m",
            "amdsharktank.examples.export_paged_llm_v1",
            f"--irpa-file={self.irpa_path}",
            f"--output-mlir={self.output_mlir}",
            f"--output-config={self.output_config}",
            f"--bs-prefill={batch_size}",
            f"--bs-decode={batch_size}",
            f"--block-seq-stride={self.block_seq_stride}",
            f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            f"--pipeline-parallelism-size={self.pipeline_parallelism_size}",
        ]

        if self.attention_kernel is not None:
            export_args.append(f"--attention-kernel={self.attention_kernel}")
        if self.matmul_kernel is not None:
            export_args.append(f"--matmul-kernel='{self.matmul_kernel}'")

        if self.attention_dtype is not None:
            export_args.append(f"--attention-dtype={self.attention_dtype}")
        if self.activation_dtype is not None:
            export_args.append(f"--activation-dtype={self.activation_dtype}")
        if self.kv_cache_dtype is not None:
            export_args.append(f"--kv-cache-dtype={self.kv_cache_dtype}")
        if skip_decode:
            export_args.append("--skip-decode")
        if not self.interleave_rotary:
            export_args.append("--use-hf")
        if self.use_qk_norm:
            export_args.append("--use-qk-norm")
        if self.attention_chunk_size:
            export_args.append(f"--attention-chunk-size={self.attention_chunk_size}")

        self._run_cmd(
            cmd=subprocess.list2cmdline(export_args),
            run_msg="Exporting MLIR",
            success_msg="Exported to MLIR successfully",
            exception=ExportMlirException,
        )

    @timeit
    def compile_to_vmfb(
        self,
        *,
        hal_dump_path: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """
        Compiles the exported MLIR to a VMFB file.

        Raises an IreeCompileException if compilation fails for some reason.

        Args:
            hal_dump_path: Optional path where dump HAL files.
            extra_args: Additional arguments for the IREE compiler.
        """

        if self.output_vmfb is not None:
            logger.info(f" Using pre-compiled vmfb: {self.output_vmfb}")
            return
        else:
            self.output_vmfb = self.output_name.with_suffix(".vmfb")

        compile_args = [
            f"iree-compile",
            f"{self.output_mlir}",
            f"-o={self.output_vmfb}",
        ]
        compile_args += get_iree_compiler_flags_from_object(
            self, device_count=self.parallelism_size
        )
        if hal_dump_path:
            compile_args += [
                f"--iree-hal-dump-executable-files-to={hal_dump_path}/files"
            ]

        compile_args += [
            "--iree-opt-level=O3",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
            "--iree-hip-encoding-layout-resolver=data-tiling",
        ]

        # TODO: https://github.com/iree-org/iree/issues/21068
        if any(
            llama_size in str(self.irpa_path) for llama_size in ["405", "70"]
        ) and all("max-iterations" not in arg for arg in compile_args):
            compile_args += ["--iree-stream-affinity-solver-max-iterations=1024"]

        # Append optional arguments if provided
        if extra_args:
            compile_args += extra_args

        self._run_cmd(
            cmd=subprocess.list2cmdline(compile_args),
            run_msg="Launching compile command",
            success_msg="Compiled to VMFB successfully",
            exception=IreeCompileException,
        )

    def iree_benchmark(
        self,
        *,
        benchmark_filename: Optional[Path] = None,
        extra_args: List[str],
    ) -> None:
        """
        Runs a compiled program with the given args using `iree-benchmark-module`.
        This assumes that the `iree-benchmark-module` command is available (usually via PATH).

        Raises an IreeBenchmarkException if running fails for some reason.

        Args:
            benchmark_filename: Optional path to the benchmark file.
            extra_args: List of arguments to pass to `iree-benchmark-module`.
            compile_cmd: Command used to compile the program, for inclusion in error messages.
        """
        params, devices, rocr_visible_devices = self._prepare_params_and_devices()

        benchmark_args = [
            *rocr_visible_devices,
            "iree-benchmark-module",
            "--hip_use_streams=true",
            f"--module={self.output_vmfb}",
            *params,
            *devices,
            *extra_args,
            str(benchmark_filename),
        ]

        self._run_cmd(
            cmd=subprocess.list2cmdline(benchmark_args),
            run_msg="Launching benchmark command",
            success_msg="Benchmarked successfully",
            exception=IreeBenchmarkException,
        )

    @timeit
    def iree_run(
        self,
        *,
        extra_args: List[str],
        output_paths: Optional[list[str | Path]] = None,
    ) -> list[torch.Tensor]:
        """
        Run the compiled module using `iree-run-module` with the specified HIP device ID and additional arguments.

        Raises an IreeRunException if running fails for some reason.

        Args:
            extra_args: Additional arguments to pass to the `iree-run-module` command.
            output_paths: List of paths to save the output tensors. If None, or empty list, no outputs are saved.

        Returns:
            A list of torch tensors loaded from the output paths, if any.
        """
        if output_paths is None:
            output_paths = []

        output_paths = [
            Path(path).resolve().with_suffix(".npy") for path in output_paths
        ]
        output_path_args = [f"--output=@{path}" for path in output_paths]

        params, devices, rocr_visible_devices = self._prepare_params_and_devices()
        run_args = [
            *rocr_visible_devices,
            "iree-run-module",
            "--hip_use_streams=true",
            f"--module={self.output_vmfb}",
            *params,
            *devices,
            *extra_args,
            *output_path_args,
        ]
        self._run_cmd(
            cmd=subprocess.list2cmdline(run_args),
            run_msg="Launching run command",
            success_msg="Run completed successfully",
            exception=IreeRunException,
        )

        return [torch.from_numpy(np.load(path)) for path in output_paths]

    def export_and_compile_llm(
        self,
        *,
        batch_size: int,
        skip_decode: bool = False,
        hal_dump_path: Optional[Path] = None,
        extra_compile_args: Optional[List[str]] = None,
    ) -> str:
        """
        Helper function to export the LLM to MLIR and compile it to VMFB in one call.

        Args:
            batch_size: The batch size to use for prefill and decode.
            skip_decode: If True, skips the decoding step during export.
            hal_dump_path: Optional path where dump HAL files.
            extra_compile_args: Additional arguments for the IREE compiler.

        Returns:
            The path to the compiled VMFB file as a string.
        """
        if self.output_vmfb is not None:
            logger.info(f" Using pre-compiled vmfb: {self.output_vmfb}")
            return str(Path(self.output_vmfb).resolve())

        self.export_llm_to_mlir(batch_size=batch_size, skip_decode=skip_decode)
        self.compile_to_vmfb(extra_args=extra_compile_args, hal_dump_path=hal_dump_path)
        return str(Path(self.output_vmfb).resolve())
