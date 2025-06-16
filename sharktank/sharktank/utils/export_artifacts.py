# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from datetime import timedelta
from typing import Any, List, Optional, TYPE_CHECKING
from sharktank.utils.iree import get_iree_compiler_flags_from_object

if TYPE_CHECKING:
    from sharktank.layers import LlamaModelConfig

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

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
    """shark-ai export MLIR exception."""

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
    def __init__(
        self,
        *,
        irpa_path: str,
        batch_size: int,
        attention_kernel: str,
        tensor_parallelism_size: int,
        pipeline_parallelism_size: int,
        block_seq_stride: int,
        iree_hal_target_device: str,
        iree_hip_target: str | None = None,
        iree_hal_local_target_device_backends: str | None = None,
        use_attention_mask: bool = False,
        use_hf: bool = False,
        activation_dtype: str = "float16",
        attention_dtype: str = "float16",
        kv_cache_dtype: Optional[str] = None,
        output_mlir: Optional[str] = None,
        output_config: Optional[str] = None,
    ):
        self.sharktank_dir = str(
            Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
        )
        self.irpa_path = irpa_path
        self.output_mlir = output_mlir
        self.output_config = output_config
        self.batch_size = batch_size
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.iree_hal_local_target_device_backends = (
            iree_hal_local_target_device_backends
        )
        self.attention_kernel = attention_kernel
        self.tensor_parallelism_size = tensor_parallelism_size
        self.pipeline_parallelism_size = pipeline_parallelism_size
        self.parallelism_size = (
            self.tensor_parallelism_size * self.pipeline_parallelism_size
        )
        self.block_seq_stride = block_seq_stride
        self.use_attention_mask = use_attention_mask
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.use_hf = use_hf

    @staticmethod
    def from_config(
        config: "LlamaModelConfig", /, **init_kwargs: dict[str, Any]
    ) -> "ExportArtifacts":
        properties = config.to_properties()
        kv_cache_dtype = (
            properties["kv_cache_dtype"] if "kv_cache_dtype" in properties else None
        )
        return ExportArtifacts(
            attention_kernel=config.attention_kernel,
            tensor_parallelism_size=config.tensor_parallelism_size,
            pipeline_parallelism_size=config.pipeline_parallelism_size,
            block_seq_stride=config.block_seq_stride,
            use_hf=config.use_hf,
            activation_dtype=properties["activation_dtype"],
            attention_dtype=properties["attention_dtype"],
            kv_cache_dtype=kv_cache_dtype,
            **init_kwargs,
        )

    def _run_cmd(
        self,
        cmd: str,
        cwd: str,
        run_msg: str,
        success_msg: str,
        exception: ExportArtifactsException,
    ):
        """Helper function to run a command and handle exceptions."""
        logger.info(f"{run_msg}:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise exception(proc, cwd)
        else:
            logger.info(f"{success_msg}:\n" f"{proc.stdout}")

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            total_seconds = end - start
            time_taken = str(abs(timedelta(seconds=total_seconds)))
            hours, minutes, seconds = time_taken.split(":")

            if total_seconds < 1:
                time_taken = f" {round(total_seconds * 1000, 3)} ms"
            elif total_seconds < 60:
                time_taken = "{:.2f} secs".format(round(float(total_seconds), 2))
            else:
                time_taken = "{:02d} hrs : {:02d} mins : {:.2f} secs".format(
                    int(hours), int(minutes), round(float(seconds), 2)
                )

            func_name = func.__name__
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

    @timeit
    def shard_irpa_file(
        self,
        *,
        irpa_file: str,
        output_irpa: str,
    ):
        shard_irpa_args = [
            "python3",
            "-m",
            "sharktank.examples.sharding.shard_llm_dataset",
            "--irpa-file",
            irpa_file,
            "--output-irpa-file",
            output_irpa,
            "--tensor-parallelism-size",
            str(self.tensor_parallelism_size),
        ]

        self._run_cmd(
            cmd=subprocess.list2cmdline(shard_irpa_args),
            cwd=self.sharktank_dir,
            run_msg="Sharding irpa file",
            success_msg="Sharded irpa file successfully",
            exception=IrpaShardException,
        )

    @timeit
    def export_to_mlir(
        self,
        *,
        output_mlir: str,
        output_config: str,
        skip_decode: Optional[bool] = None,
    ):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            f"--irpa-file={self.irpa_path}",
            f"--output-mlir={output_mlir}",
            f"--output-config={output_config}",
            f"--bs-prefill={str(self.batch_size)}",
            f"--bs-decode={str(self.batch_size)}",
            f"--block-seq-stride={self.block_seq_stride}",
            f"--attention-dtype={self.attention_dtype}",
            f"--activation-dtype={self.activation_dtype}",
            f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            f"--pipeline-parallelism-size={self.pipeline_parallelism_size}",
        ]

        # TODO: This check should be handled by the export script.
        assert self.attention_kernel in [
            "decomposed",
            "torch",
            "sharktank",
        ], "Only torch (sdpa), decomposed or sharktank --attention-kernel types are supported"

        export_args.append(f"--attention-kernel={self.attention_kernel}")

        if self.kv_cache_dtype is not None:
            export_args.append(f"--kv-cache-dtype={self.kv_cache_dtype}")
        if skip_decode:
            export_args.append("--skip-decode")
        if self.use_attention_mask:
            export_args.append("--use-attention-mask")
        if self.use_hf:
            export_args.append("--use-hf")

        self._run_cmd(
            cmd=subprocess.list2cmdline(export_args),
            cwd=self.sharktank_dir,
            run_msg="Exporting MLIR",
            success_msg="Exported to MLIR successfully",
            exception=ExportMlirException,
        )

    @timeit
    def compile_to_vmfb(
        self,
        *,
        output_mlir,
        output_vmfb,
        cwd: str | None = None,
        hal_dump_path: Optional[Path] = None,
        args: Optional[List[str]] = None,
    ):
        if cwd is None:
            cwd = os.getcwd()

        # TODO: Control flag to enable multiple backends
        compile_args = [
            f"iree-compile",
            f"{output_mlir}",
            f"-o={output_vmfb}",
        ]
        compile_args += get_iree_compiler_flags_from_object(
            self, device_count=self.parallelism_size
        )
        if hal_dump_path:
            compile_args += [
                f"--iree-hal-dump-executable-files-to={hal_dump_path}/files"
            ]
        # Append optional arguments if provided
        if args:
            compile_args += args
        else:
            compile_args += [
                "--iree-opt-level=O3",
                "--iree-hal-indirect-command-buffers=true",
                "--iree-stream-resource-memory-model=discrete",
                "--iree-hal-memoization=true",
            ]

        self._run_cmd(
            cmd=subprocess.list2cmdline(compile_args),
            cwd=cwd,
            run_msg="Launching compile command",
            success_msg="Compiled MLIR successfully",
            exception=IreeCompileException,
        )

    def iree_benchmark_vmfb(
        self,
        *,
        hip_device_id: str,
        vmfb_name: str,
        irpa_path: str,
        benchmark_filename: Optional[Path] = None,
        args: List[str],
        cwd: str | Path,
    ):
        """Runs a compiled program with the given args using `iree-benchmark-module`.
        This assumes that the `iree-benchmark-module` command is available (usually via PATH).
        Args:
            vmfb_name: Name of the .vmfb file (relative to `cwd`).
            args: List of arguments to pass to `iree-benchmark-module`.
            cwd: Working directory to run the command within. (either string or Path works)
            compile_cmd: Command used to compile the program, for inclusion in error messages.
        Raises Exception if running fails for some reason.
        """
        benchmark_args = []
        if self.parallelism_size > 1:
            base_irpa_path, _ = os.path.splitext(irpa_path)
            rocr_visible_devices = [
                f"ROCR_VISIBLE_DEVICES={','.join(str(i) for i in range(self.parallelism_size))}"
            ]
            params = [f"--parameters=model={base_irpa_path}.irpa"]
            params += [
                f"--parameters=model={base_irpa_path}.rank{i}.irpa"
                for i in range(self.tensor_parallelism_size)
            ]
            devices = [f"--device=hip://{i}" for i in range(self.parallelism_size)]
        else:
            hip_device_arg = int(hip_device_id.split("://")[1])
            rocr_visible_devices = [
                f"ROCR_VISIBLE_DEVICES={','.join(str(i) for i in range(hip_device_arg + 1))}"
            ]
            params = [f"--parameters=model={irpa_path}"]
            devices = [f"--device={hip_device_id}"]
        benchmark_args += rocr_visible_devices
        benchmark_args += [
            "iree-benchmark-module",
            "--hip_use_streams=true",
            f"--module={vmfb_name}",
        ]
        benchmark_args += params
        benchmark_args += devices
        benchmark_args += args
        benchmark_args += [str(benchmark_filename)]

        self._run_cmd(
            cmd=subprocess.list2cmdline(benchmark_args),
            cwd=str(cwd),
            run_msg="Launching benchmark command",
            success_msg="Benchmarked successfully",
            exception=IreeBenchmarkException,
        )

    def create_file(self, *, suffix, prefix):
        # TODO: This looks scary. Should not be doing an fopen just to ensure the path exists, who closes this?
        file_path = Path(prefix).with_suffix(suffix)
        f = open(file_path, "w")
        return file_path

    def get_artifacts(self):

        self.dir_path = (
            self.sharktank_dir + "/" + "perplexity_ci_artifacts/"
        )  # TODO: Remove this hardcoded path
        temp_dir = Path(self.dir_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        model_name = (
            str(self.irpa_path).split("/")[-1].rsplit(".", 1)[0].replace(".", "_")
            + "_"
            + self.attention_kernel
            + (
                f"_pp{self.pipeline_parallelism_size}"
                if self.pipeline_parallelism_size > 1
                else ""
            )
        )

        if self.output_mlir is None:
            self.output_mlir = str(
                self.create_file(suffix=".mlir", prefix=self.dir_path + model_name)
            )
            self.output_config = str(
                self.create_file(suffix=".json", prefix=self.dir_path + model_name)
            )

            self.export_to_mlir(
                output_mlir=self.output_mlir,
                output_config=self.output_config,
            )
        else:
            logger.info(f" Using pre-exported mlir: {self.output_mlir}")
            logger.info(f" Using pre-exported config json: {self.output_config}")

        output_vmfb = str(
            self.create_file(suffix=".vmfb", prefix=self.dir_path + model_name)
        )

        self.compile_to_vmfb(
            output_mlir=self.output_mlir,
            output_vmfb=output_vmfb,
            cwd=self.sharktank_dir,
        )

        return output_vmfb
