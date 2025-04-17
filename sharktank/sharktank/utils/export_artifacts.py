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
from typing import List, Optional

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)


class ExportMlirException(Exception):
    """shark-ai export MLIR exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        super().__init__(
            f"Error invoking export_paged_llama_v1.py\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        super().__init__(
            f"Error invoking iree-compile\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


class IreeBenchmarkException(Exception):
    """Runtime exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        # iree-run-module sends output to both stdout and stderr
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)
        super().__init__(
            f"Error invoking iree-benchmark-module\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n"
            f"Stdout diagnostics:\n{outs}\n"
            f"Run with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


class ExportArtifacts:
    def __init__(
        self,
        *,
        irpa_path: str,
        batch_size: int,
        iree_hip_target: str,
        attention_kernel: str,
        tensor_parallelism_size: int,
        block_seq_stride: int,
        iree_hal_target_device: str,
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
        self.attention_kernel = attention_kernel
        self.tensor_parallelism_size = tensor_parallelism_size
        self.block_seq_stride = block_seq_stride
        self.use_attention_mask = use_attention_mask
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.use_hf = use_hf

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

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(shard_irpa_args)

        logger.info(f"Sharding irpa file:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd, text=True)
        if proc.returncode != 0:
            logger.error(
                f"Error sharding irpa file with shard_llm_dataset.py\n"
                f"{proc.stdout+proc.stderr}"
            )
        else:
            logger.info(f"Sharded irpa file successfully:\n" f"{proc.stdout}")

        return proc.returncode

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
        ]

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

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(export_args)

        logger.info(f" Exporting mlir:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd, text=True)
        if proc.returncode != 0:
            raise ExportMlirException(proc, cwd)
        else:
            logger.info(f" Exported to mlir successfully:\n" f"{proc.stdout}")

        return proc.returncode

    @timeit
    def compile_to_vmfb(
        self,
        *,
        output_mlir,
        output_vmfb,
        cwd,
        hal_dump_path: Optional[Path] = None,
        args: Optional[List[str]] = None,
    ):

        # TODO: Control flag to enable multiple backends
        compile_args = [
            f"iree-compile",
            f"{output_mlir}",
            f"--iree-hip-target={self.iree_hip_target}",
            f"-o={output_vmfb}",
        ]
        if self.tensor_parallelism_size > 1:
            iree_hal_target_devices = [
                f"--iree-hal-target-device={self.iree_hal_target_device}[{i}]"
                for i in range(self.tensor_parallelism_size)
            ]
        else:
            iree_hal_target_devices = [
                f"--iree-hal-target-device={self.iree_hal_target_device}"
            ]
        compile_args += iree_hal_target_devices
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

        cmd = subprocess.list2cmdline(compile_args)

        logger.info(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeCompileException(proc, cwd)

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
        if self.tensor_parallelism_size > 1:
            base_irpa_path, _ = os.path.splitext(irpa_path)
            rocr_visible_devices = [
                f"ROCR_VISIBLE_DEVICES={','.join(str(i) for i in range(self.tensor_parallelism_size))}"
            ]
            params = [f"--parameters=model={base_irpa_path}.irpa"]
            params += [
                f"--parameters=model={base_irpa_path}.rank{i}.irpa"
                for i in range(self.tensor_parallelism_size)
            ]
            devices = [
                f"--device=hip://{i}" for i in range(self.tensor_parallelism_size)
            ]
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
        cmd = subprocess.list2cmdline(benchmark_args)
        logger.info(f" Launching run command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=sys.stdout, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeBenchmarkException(proc, cwd)

    def create_file(self, *, suffix, prefix):
        file_path = Path(prefix).with_suffix(suffix)
        f = open(file_path, "w")
        return file_path

    def get_artifacts(self):

        self.dir_path = self.sharktank_dir + "/" + "perplexity_ci_artifacts/"
        temp_dir = Path(self.dir_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        model_name = (
            str(self.irpa_path).split("/")[-1].rsplit(".", 1)[0].replace(".", "_")
            + "_"
            + self.attention_kernel
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
