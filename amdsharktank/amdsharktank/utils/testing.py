# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import contextlib
from pathlib import Path
import numpy as np
import pytest
from os import PathLike
import functools
import os
import re
import shutil
import tempfile
import unittest
from typing import Any, Callable
from operator import eq
from collections.abc import Iterable
import gc
import random
import torch
import inspect
from dataclasses import dataclass, field

from sys import platform
from datasets import load_dataset

from amdsharktank.types import *
from amdsharktank.types.pipelining import pipeline_parallelize_llm_theta
from amdsharktank.utils.io import ShardedArchiveBuilder
from .math import cosine_similarity
from amdsharktank.ops.utils import get_all_implementations, cast_to_type_spec
from amdsharktank.ops._registry import _matches

# TODO: ci-amdsharktank-nightly should run all nightly CIs and ci-amdsharktank/test-mi300x should run all pre-submits
# requiring mi300x in a single workflow, dropping all test specific flags/workflows
is_pre_submit = pytest.mark.skipif(
    'not config.getoption("run-quick-test")',
    reason="Run quick tests if --run-quick-test is passed",
)
is_nightly = pytest.mark.skipif(
    'not config.getoption("run-nightly-test")',
    reason="Run large tests if --run-nightly-test is passed",
)
is_sharded = pytest.mark.skipif(
    'not config.getoption("run-sharded-test")',
    reason="Run sharded tests if --run-sharded-test is passed",
)
is_llama_8b = pytest.mark.skipif(
    'config.getoption("llama3_8b_f16_model_path") is None',
    reason="Run llama tests if --llama3-8b-f16-model-path is passed",
)
is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")
is_mi350x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx950'")
is_cpu_condition = (
    "exec('from amdsharktank.utils.testing import is_iree_hal_target_device_cpu') or "
    "is_iree_hal_target_device_cpu(config.getoption('iree_hal_target_device'))"
)
is_not_cpu_condition = (
    "exec('from amdsharktank.utils.testing import is_iree_hal_target_device_cpu') or "
    "not is_iree_hal_target_device_cpu(config.getoption('iree_hal_target_device'))"
)
is_hip_condition = "config.getoption('iree_hal_target_device') == 'hip'"
is_cpu = pytest.mark.skipif(is_not_cpu_condition)
is_cpu_win = pytest.mark.skipif(is_cpu_condition and platform == "win32")


@dataclass
class IreeFlags:
    iree_device: str
    iree_hip_target: str
    iree_hal_target_device: str
    iree_hal_local_target_device_backends: str


def is_iree_hal_target_device_cpu(v: str, /) -> bool:
    return v.startswith("local") or v == "llvm-cpu"


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._temp_dir = Path(tempfile.mkdtemp(type(self).__qualname__))

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self._temp_dir, ignore_errors=True)


class MainRunnerTestBase(TempDirTestBase):
    """Performs an in-process test of a `main(args)` func."""

    def get_file_path(self, name: str) -> Path:
        return self._temp_dir / name

    def get_irpa_path(self, name: str) -> Path:
        return self.get_file_path(f"{name}.irpa")

    def save_dataset(self, ds: Dataset, name: str) -> Path:
        p = self.get_irpa_path(name)
        ds.save(p)
        return p

    def run_main(self, main_func, *args):
        new_args = [str(arg) for arg in args]
        main_func(new_args)

    def assertFileWritten(self, p: Path):
        self.assertTrue(p.exists(), msg=f"Expected file {p} was not created")
        self.assertGreater(p.stat().st_size, 0, msg=f"Expected file {p} had zero size")


class IreeVsEagerLLMTester:
    """
    Class for comparing the results of IREE and eager execution of the same LLM.

    Can only be run on with a gpu enabled version of torch.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        theta: Theta,
        config: "LlamaModelConfig",
        torch_device: str,
        iree_device: str,
        iree_hip_target: str,
        iree_hal_target_device: str,
        raw_token_ids: list[list[int]] | None = None,
        skip_decode: bool = False,
        use_qk_norm: bool = False,
        attention_chunk_size: Optional[int] = None,
    ):

        """
        Setup the variables and objectes needed for the IREE vs eager test.

        Args:
            work_dir: The directory to save the results and intermediate files.
            theta: The Theta object containing the model parameters.
            config: The configuration for the model.
            torch_device: The device to use for the eager execution (e.g., "cuda:0" or "cpu").
            iree_device: The IREE device to use for IREE execution.
            iree_hip_target: The IREE HIP target to use for IREE execution (e.g, "gfx942").
            iree_hal_target_device: The IREE HAL target device to use for IREE execution (e.g., "hip" or "llvm-cpu").
            raw_token_ids: The raw token ids to use for the prefill stage. If none are provided, a static set will be generated.
            skip_decode: Whether to skip the decode stage. If True, the decode stage will not be run, and the decode results will not be compared.
            use_qk_norm: whether to normalize q and k in the attention layer
            attention_chunk_size: size of chunk of attentions
        """
        # Note: Here to prevent circular imports
        from amdsharktank.models.llm.llm import PagedLlmModelV1
        from amdsharktank.utils.evaluate import pad_tokens
        from amdsharktank.utils.load_llm import TorchGenerator
        from amdsharktank.utils.export_artifacts import ExportArtifacts

        work_dir = work_dir if work_dir else self._temp_dir

        if raw_token_ids is None:
            raw_token_ids = self.generate_raw_token_ids()

        self.config = config
        self.skip_decode = skip_decode

        parallelism_size = (
            self.config.tensor_parallelism_size * self.config.pipeline_parallelism_size
        )
        rank_shard = ".rank0" if parallelism_size > 1 else ""
        rank_pipeline = ""
        if self.config.pipeline_parallelism_size > 1:
            rank_pipeline = "_0"

        # NOTE: These paths must match those used by load_llm.py::Batch when it dumps the values
        self.prefill_iree_logits_path = [work_dir / "prefill_iree_logits.npy"]
        self.prefill_eager_logits_path = work_dir / "prefill_eager_logits.npy"
        self.prefill_token_ids_path = work_dir / f"prefill_token_ids{rank_shard}.npy"
        self.prefill_seq_lens_path = work_dir / "prefill_seq_lens.npy"
        self.prefill_seq_block_ids_path = (
            work_dir / f"prefill_seq_block_ids{rank_pipeline}{rank_shard}.npy"
        )

        # prefill_token_ids_0.rank0.npy
        # prefill_token_ids.rank0.npy

        # NOTE: The _0 is required to match load_llm.py::Batch's syntax.
        self.decode_iree_results_path = [work_dir / "decode_iree_logits.npy"]
        self.decode_eager_logits_path = work_dir / "decode_eager_logits_0.npy"
        self.decode_next_tokens_path = (
            work_dir / f"decode_next_tokens_0{rank_shard}.npy"
        )
        self.decode_seq_lens_path = work_dir / "decode_seq_lens_0.npy"
        self.decode_seq_block_ids_path = (
            work_dir / f"decode_seq_block_ids{rank_pipeline}_0{rank_shard}.npy"
        )
        self.decode_start_positions_path = (
            work_dir / f"decode_start_positions{rank_pipeline}_0{rank_shard}.npy"
        )

        self.prefill_cache_state_paths = []
        self.decode_cache_state_paths = []
        if parallelism_size == 1:
            # If we are not using parallelism, we only need one cache state file
            self.prefill_cache_state_paths.append(work_dir / "prefill_cache_state.npy")
            self.decode_cache_state_paths.append(work_dir / "decode_cache_state_0.npy")
        else:
            for pp in range(self.config.pipeline_parallelism_size):
                for tp in range(self.config.tensor_parallelism_size):
                    tp_rank = f".rank{tp}"
                    pp_rank = ""
                    if config.pipeline_parallelism_size > 1:
                        pp_rank = f"_{pp}"
                    prefill_name = f"prefill_cache_state{pp_rank}{tp_rank}.npy"
                    decode_name = f"decode_cache_state{pp_rank}_0{tp_rank}.npy"
                    self.prefill_cache_state_paths.append(work_dir / prefill_name)
                    self.decode_cache_state_paths.append(work_dir / decode_name)

        self.dataset_path = work_dir / "parameters.irpa"

        if self.config.tensor_parallelism_size > 1:
            from amdsharktank.types.sharding import shard_theta

            theta = shard_theta(theta=theta, config=config)

        Dataset(root_theta=theta, properties=self.config.to_properties()).save(
            path=self.dataset_path
        )
        self.exporter = ExportArtifacts.from_config(
            self.config,
            irpa_path=self.dataset_path,
            iree_hip_target=iree_hip_target,
            iree_hal_target_device=iree_hal_target_device,
            hip_device_id=iree_device,
            output_name=work_dir / "model",
            use_qk_norm=use_qk_norm,
            attention_chunk_size=attention_chunk_size,
        )

        # Note: Must be after saving the dataset and creating the exporter but before moving theta to the provided device.
        pipeline_parallelize_llm_theta(theta, self.config.parallelism_config)

        self.config.device = torch.device(torch_device)  # Switch to gpu for eager mode
        theta_for_eager = theta.to(device=self.config.device)

        prefill_token_ids, prefill_seq_lens = pad_tokens(
            token_ids=raw_token_ids,
            pad_to_multiple_of=self.config.block_seq_stride,
        )
        prefill_token_ids = torch.as_tensor(
            prefill_token_ids, device=self.config.device
        )
        prefill_seq_lens = torch.as_tensor(prefill_seq_lens, device=self.config.device)
        self.batch_size = prefill_token_ids.shape[0]

        generator = TorchGenerator(
            PagedLlmModelV1(theta=theta_for_eager, config=self.config)
        )
        self.eager_batch = generator.begin_batch(
            token_ids=prefill_token_ids,
            seq_lens=prefill_seq_lens,
            dump_path=work_dir,
        )

        self.exporter.export_and_compile_llm(
            batch_size=self.batch_size, skip_decode=self.skip_decode
        )

    def compare_outputs(
        self,
        *,
        eager_result: torch.Tensor,
        iree_result: torch.Tensor,
        stage_name: str,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> None:
        """
        Compare the iree results with the eager results, one sequence at a time.

        Args:
            eager_result: The result from the eager execution
            iree_result: The result from the IREE execution
            stage_name: The name of the stage being compared (e.g., "prefill", "decode")
            rtol: Relative tolerance for the comparison
            atol: Absolute tolerance for the comparison
        """
        for i, (eager_i, iree_i) in enumerate(zip(eager_result, iree_result)):
            try:
                torch.testing.assert_close(
                    actual=iree_i, expected=eager_i, rtol=rtol, atol=atol
                )
            except AssertionError as error:
                raise AssertionError(
                    f"Outputs do not match for {stage_name} batch index {i}:\n"
                    f"Eager: {eager_i}\n"
                    f"IREE: {iree_i}\n"
                ) from error

    def generate_raw_token_ids(self):
        """
        Generate a static set of raw token ids to use for the prefill stage.
        """
        # Use a fixed set of prompts for testing
        return [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
            [3, 5, 2, 1],
            [3, 5, 2, 1, 5],  # Adding a longer sequence to test padding
        ]

    def run_and_compare_iree_vs_eager(
        self, *, rtol: float | None = None, atol: float | None = None
    ):
        """
        Run the IREE and eager execution and compare the outputs.
        If comparison passes"""
        self.run_eager()
        self.run_iree()
        self.compare_outputs(
            eager_result=self.eager_prefill_logits,
            iree_result=self.iree_prefill_logits,
            stage_name="prefill",
            rtol=rtol,
            atol=atol,
        )
        if not self.skip_decode:
            self.compare_outputs(
                eager_result=self.eager_decode_logits,
                iree_result=self.iree_decode_logits,
                stage_name="decode",
                rtol=rtol,
                atol=atol,
            )

    def run_eager(self):
        """
        Run the eager execution of the LLM prefill and decode stages.
        Saveing the cache state before each stage to be used in IREE execution.
        """
        eager_decode_tokens = self.eager_batch.prefill()
        self.eager_batch.dump_args(
            phase="prefill",
            arg_name="eager_logits",
            arg=self.eager_batch.prefill_logits,
        )

        self.eager_prefill_logits = torch.tensor(
            np.load(self.prefill_eager_logits_path)
        )

        if not self.skip_decode:
            self.eager_batch.decode(token_batch=eager_decode_tokens)
            self.eager_batch.dump_args(
                phase="decode",
                arg_name="eager_logits",
                arg=self.eager_batch.prefill_logits,
                decode_step=0,
            )

            self.eager_decode_logits = torch.tensor(
                np.load(self.decode_eager_logits_path)
            )

        self.config.device = torch.device("cpu")  # Switch back to cpu for tracing

    def run_iree(self):
        """
        Run the iree execution using the inputs from the eager execution.
        """
        prefill_args = [
            f"--function=prefill_bs{self.batch_size}",
            f"--input=@{self.prefill_token_ids_path}",
            f"--input=@{self.prefill_seq_lens_path}",
            f"--input=@{self.prefill_seq_block_ids_path}",
            *(f"--input=@{path}" for path in self.prefill_cache_state_paths),
        ]
        self.iree_prefill_logits = self.exporter.iree_run(
            extra_args=prefill_args,
            output_paths=self.prefill_iree_logits_path,
        )[0]

        if not self.skip_decode:
            decode_args = [
                f"--function=decode_bs{self.batch_size}",
                f"--input=@{self.decode_next_tokens_path}",
                f"--input=@{self.decode_seq_lens_path}",
                f"--input=@{self.decode_start_positions_path}",
                f"--input=@{self.decode_seq_block_ids_path}",
                *(f"--input=@{path}" for path in self.decode_cache_state_paths),
            ]
            self.iree_decode_logits = self.exporter.iree_run(
                extra_args=decode_args,
                output_paths=self.decode_iree_results_path,
            )[0]


@contextlib.contextmanager
def temporary_directory(identifier: str):
    """Returns a context manager TemporaryDirectory suitable for testing.

    If the env var amdsharkTANK_TEST_ASSETS_DIR is set then directories will be
    created under there, named by `identifier`. If the `identifier` subdirectory
    exists, it will be deleted first.

    This is useful for getting updated goldens and such.
    """
    explicit_dir = os.getenv("amdsharkTANK_TEST_ASSETS_DIR", None)
    if explicit_dir is None:
        with tempfile.TemporaryDirectory(prefix=f"{identifier}_") as td:
            yield td
    else:
        explicit_path = Path(explicit_dir) / identifier
        if explicit_path.exists():
            shutil.rmtree(explicit_path)
        explicit_path.mkdir(parents=True, exist_ok=True)
        yield explicit_path


@contextlib.contextmanager
def override_debug_flags(flag_updates: dict):
    from .debugging import flags

    restore = {}
    try:
        for k, v in flag_updates.items():
            print(f"Overriding debug flag {k} = {v}")
            current_value = getattr(flags, k)
            restore[k] = current_value
            setattr(flags, k, v)
        yield
    finally:
        for k, v in restore.items():
            print(f"Restoring debug flag {k} = {v}")
            setattr(flags, k, v)


def get_best_torch_device() -> str:
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"


def assert_dicts_equal(
    dict1: dict, dict2: dict, *, values_equal: Callable[[Any, Any], bool] | None = None
) -> None:
    values_equal = values_equal or eq
    assert len(dict1) == len(
        dict2
    ), f"Dictionaries not equal. {dict1} and {dict2} have different number of elements {len(dict1)} != {len(dict2)}"
    for k, v1 in dict1.items():
        assert (
            k in dict2
        ), f"Dictionaries {dict1} and {dict2} not equal. Key {k} not found in {dict2}"
        v2 = dict2[k]
        assert values_equal(
            v1, dict2[k]
        ), f"Dictionaries {dict1} and {dict2} not equal for key {k}. Values {v1} and {v2} not equal"


def assert_equal(
    a: Any, b: Any, *, equal: Callable[[Any, Any], bool] | None = None
) -> None:
    equal = equal or eq
    assert equal(a, b), f"{a} and {b} are not equal"


def assert_close_safetensors(
    actual_path: PathLike,
    ref_path: PathLike,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    fail_fast: bool = True,
    check_dtype: bool = True,
):
    """Asserts that actual and reference safetensors files are within tolerances.

    actual_path and ref_path can be directories. In that case files with matching
    sub-paths will be compared."""
    from safetensors import safe_open
    import torch

    print(f"Asserting tensors close: actual={actual_path}, ref={ref_path}")

    ref_path = Path(ref_path)
    actual_path = Path(actual_path)

    assert ref_path.exists(), f'Path "{ref_path}" not found'

    if not ref_path.is_file():
        # Get all files in ref_path recursively.
        ref_file_paths: list[Path] = [
            file_path
            for file_path in Path(ref_path).rglob("*.safetensors")
            if file_path.is_file()
        ]

        # Sort by timestamp. When we compare traces we want to order by time.
        ref_file_paths.sort(key=lambda file_path: os.stat(file_path).st_mtime_ns)

        ref_actual_file_path_map: dict[Path, Path] = {
            ref_file_path: Path(actual_path) / ref_file_path.relative_to(ref_path)
            for ref_file_path in ref_file_paths
        }

        not_close_list: list[tuple[Path, Path]] = []
        for ref_file_path, actual_file_path in ref_actual_file_path_map.items():
            try:
                assert os.path.isfile(actual_file_path)
                assert_close_safetensors(
                    actual_file_path,
                    ref_file_path,
                    rtol=rtol,
                    atol=atol,
                    fail_fast=fail_fast,
                    check_dtype=check_dtype,
                )
            except Exception as ex:
                if fail_fast:
                    raise
                not_close_list.append((actual_file_path, ref_file_path))

        if len(not_close_list) > 0:
            print("Not close:")
            for actual, ref in not_close_list:
                print(f"{actual} != {ref}")
            assert False, "Tensors are not close."
        return

    def print_stats(label: str, t: torch.Tensor):
        t_f32 = t.to(dtype=torch.float32)
        std, mean = torch.std_mean(t_f32)
        print(
            f"    {label}: "
            f"MIN={torch.min(t_f32)}, "
            f"MAX={torch.max(t_f32)}, "
            f"MEAN={mean}, STD={std}, "
            f"DTYPE={t.dtype}"
        )

    with safe_open(actual_path, framework="pt") as actual_f, safe_open(
        ref_path, framework="pt"
    ) as ref_f:
        # Print all first.
        for name in ref_f.keys():
            actual = actual_f.get_tensor(name)
            ref = ref_f.get_tensor(name)

            print(f":: Comparing tensor {name}")
            print_stats(" REF", ref)
            print_stats(" ACT", actual)
            print_stats("DIFF", (ref - actual))
        # Then assert.
        for name in ref_f.keys():
            actual = actual_f.get_tensor(name)
            ref = ref_f.get_tensor(name)
            try:
                torch.testing.assert_close(
                    actual, ref, rtol=rtol, atol=atol, check_dtype=check_dtype
                )
            except Exception as ex:
                if fail_fast:
                    raise
                print(ex)


def assert_iterables_equal(
    iterable1: Iterable,
    iterable2: Iterable,
    *,
    elements_equal: Callable[[Any, Any], bool] | None = None,
) -> None:
    non_existent_value = object()
    elements_equal = elements_equal or eq

    def assert_elements_equal_fn(i: int, x: Any, y: Any):
        assert not (
            x is non_existent_value or y is non_existent_value
        ), f"Iterables with different size not equal at index {i}"
        assert elements_equal(
            x, y
        ), f"Iterables not equal at index {i} for elements {x} and {y}"

    for i, (v1, v2) in enumerate(zip(iterable1, iterable2, strict=True)):
        assert_elements_equal_fn(i, v1, v2)


def assert_tensor_close(
    actual: AnyTensorTree,
    expected: AnyTensorTree,
    rtol: float | None = None,
    atol: float | None = None,
    max_outliers_fraction: Optional[float] = None,
    inlier_atol: Optional[float] = None,
):
    if (max_outliers_fraction is None and inlier_atol is not None) or (
        max_outliers_fraction is not None and inlier_atol is None
    ):
        raise ValueError(
            "max_outliers_fraction and inlier_atol must be provided or not together."
        )
    # Unbox tensors.
    from amdsharktank.utils import tree

    def is_leaf(x: Any) -> bool:
        return is_any_tensor(x) or tree.is_leaf_default(x)

    actual_is_leaf = is_leaf(actual)
    expected_is_leaf = is_leaf(expected)

    assert not (
        actual_is_leaf ^ expected_is_leaf
    ), f"Actual {actual} and expected {expected} must be both leafs or both not leafs"

    if not actual_is_leaf:

        tree.assert_equal(
            actual,
            expected,
            is_leaf=is_leaf,
            assert_equal=functools.partial(
                assert_tensor_close,
                rtol=rtol,
                atol=atol,
                max_outliers_fraction=max_outliers_fraction,
                inlier_atol=inlier_atol,
            ),
        )
        return

    is_actual_tensor = is_any_tensor(actual)
    is_expected_tensor = is_any_tensor(expected)
    assert not (
        is_actual_tensor ^ is_expected_tensor
    ), f"Actual {actual} and expected {expected} must be both tensors or both not tensors"
    if not is_actual_tensor:
        return

    actual = unbox_tensor(actual)
    expected = unbox_tensor(expected)
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            check_dtype=False,
        )

        if inlier_atol is not None:
            # TODO: handle trees
            outliers = (actual - expected).abs() > inlier_atol
            outliers_fraction = outliers.count_nonzero() / outliers.numel()
            if outliers_fraction > max_outliers_fraction:
                raise AssertionError(
                    f"The fraction of outliers {outliers_fraction:%} is above the allowed "
                    f"{max_outliers_fraction:%}. Inlier atol={inlier_atol}."
                )
    except AssertionError as ex:
        from amdsharktank.ops import promote_to_float

        abs_diff = torch.abs(actual - expected)
        abs_diff_std, abs_diff_mean = torch.std_mean(promote_to_float(abs_diff))
        expected_std, expected_mean = torch.std_mean(promote_to_float(expected))
        msg = (
            "Tensors not equal.\n"
            "Absolute difference abs(actual - expected):\n"
            f"mean = {abs_diff_mean}\n"
            f"median = {abs_diff.median()}\n"
            f"std dev = {abs_diff_std}\n"
            f"min = {abs_diff.min()}\n"
            f"max = {abs_diff.max()}\n"
            "Expected:\n"
            f"mean = {expected_mean}\n"
            f"median = {expected.median()}\n"
            f"std dev = {expected_std}\n"
            f"min = {expected.min()}\n"
            f"max = {expected.max()}\n"
            f"With torch error:\n {str(ex)}\n"
        )
        raise AssertionError(msg) from ex


def assert_cosine_similarity_close(
    actual: AnyTensor,
    expected: AnyTensor,
    atol: float,
    max_outliers_fraction: Optional[float] = None,
    inlier_atol: Optional[float] = None,
    dim: int | None = None,
):
    cos_sim = cosine_similarity(
        actual,
        expected,
        dim=dim,
    )

    assert_tensor_close(
        actual=cos_sim,
        expected=torch.ones_like(cos_sim),
        rtol=0,
        atol=atol,
        max_outliers_fraction=max_outliers_fraction,
        inlier_atol=inlier_atol,
    )


def assert_text_encoder_state_close(
    actual: AnyTensor,
    expected: AnyTensor,
    atol: float,
    max_outliers_fraction: Optional[float] = None,
    inlier_atol: Optional[float] = None,
):
    """The cosine similarity has been suggested to compare encoder states.

    Dehua Peng, Zhipeng Gui, Huayi Wu -
    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold
    Effect (2023)

    shows that cosine and all Minkowski distances suffer from the curse of
    dimensionality.
    The cosine similarity ignores the vector magnitudes. We can probably come up with a
    better metric, but this is maybe good enough.

    The functions expects that the last dimension is the features per token.
    It will compute the cosine similarity for each token.
    """
    assert_cosine_similarity_close(
        actual=actual,
        expected=expected,
        atol=atol,
        max_outliers_fraction=max_outliers_fraction,
        inlier_atol=inlier_atol,
        dim=-1,
    )


def assert_logits_kl_divergence_close(
    actual: AnyTensor,
    expected: AnyTensor,
    atol: float,
):
    """
    Calculate the KL divergence loss between the actual and expected logits tensors.
    This function calculates it's own log softmax of the logits.

    Args:
        actual: The actual logits tensor.
        expected: The expected logits tensor.
        atol: The absolute tolerance for the KL divergence loss.
    """
    actual_probabilities = unbox_tensor(actual).log_softmax(dim=2, dtype=torch.float32)
    expected_probabilities = unbox_tensor(expected).log_softmax(
        dim=2, dtype=torch.float32
    )

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss = kl_loss(input=actual_probabilities, target=expected_probabilities)

    assert torch.all(
        loss.abs() <= atol
    ), f"KL divergence loss {loss} is greater than the allowed tolerance {atol}."


amdsharkTANK_TEST_SKIP_ENV_VAR = "amdsharkTANK_TEST_SKIP"


def skip(*decorator_args, **decorator_kwargs):
    """Decorator to skip a test when amdsharkTANK_TEST_SKIP env var is not set or != 0"""

    def decorator(test_item: Callable):
        if amdsharkTANK_TEST_SKIP_ENV_VAR not in os.environ:
            should_skip = True
        else:
            should_skip = os.environ[amdsharkTANK_TEST_SKIP_ENV_VAR] != "0"

        if should_skip:
            return unittest.skip(*decorator_args, **decorator_kwargs)(test_item)

        return test_item

    return decorator


def _eval_condition(c: bool | str | None) -> bool:
    if c is None:
        return True
    if isinstance(c, bool):
        return c
    raise NotImplementedError(
        "TODO: implement string condition evaluation the same way as in pytest"
    )


def get_random_test_text_prompts(
    num_prompts: int, min_prompt_length: int | None = None
):
    prompts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if min_prompt_length is not None:
        prompts = [p for p in prompts if len(p) >= min_prompt_length]
    return random.sample(prompts, num_prompts)


def get_frozen_test_text_prompts(
    num_prompts: int, min_prompt_length: int | None = None
):
    orig_rng_state = random.getstate()
    try:
        random.seed(13910398)
        return get_random_test_text_prompts(
            num_prompts=num_prompts, min_prompt_length=min_prompt_length
        )
    finally:
        random.setstate(orig_rng_state)


_test_prompts = None


def get_test_prompts():
    global _test_prompts
    if _test_prompts is None:
        _test_prompts = get_frozen_test_text_prompts(
            num_prompts=16, min_prompt_length=50
        )
    return _test_prompts


def create_sample_tensor_from_class(
    tensor_clazz: torch.Tensor.__class__
    | InferenceTensor.__class__
    | QuantizedLayout.__class__,
    shard_count: int = 2,
    base_tensor: AnyTensor | None = None,
) -> AnyTensor:
    def clone(t: AnyTensor, shard_index: int | None) -> AnyTensor:
        if isinstance(t, torch.Tensor):
            return t.clone()

        new_t = t.transform_subtensors(
            lambda _dict: {k: clone(v, shard_index) for k, v in _dict.items()}
        )
        if shard_index is not None:
            new_t.name += f".shard.{shard_index}"
        return new_t

    if base_tensor is None:
        base_tensor = torch.ones(2, 32)

    if tensor_clazz is torch.Tensor:
        return clone(unbox_tensor(base_tensor), None)

    if issubclass(tensor_clazz, DefaultPrimitiveTensor):
        return DefaultPrimitiveTensor(data=clone(unbox_tensor(base_tensor), None))

    if issubclass(tensor_clazz, BlockScaledFp4Layout):
        block_size = 32
        dtype = torch.float32
        quantizer = DynamicFp4BlockQuantizer(
            block_size=block_size, use_fe8m0_scale=True, dtype=dtype
        )
        return quantizer.quantize(base_tensor)

    if issubclass(tensor_clazz, QuantizedLayout):
        metadata = {"block_size": 1, "use_f38m0_scale": True, "signed": True}
        planes = {
            key: torch.tensor([1.0])
            for key in [
                "d",
                "qs",
                "m",
                "dmin",
                "sb_scales_high",
                "sb_scales_low",
                "sb_mins_high",
                "sb_mins_low",
            ]
        }
        layout = tensor_clazz.create(
            shape=base_tensor.shape, metadata=metadata, planes=planes
        )
        return PlanarQuantizedTensor(shape=base_tensor.shape, layout=layout)

    shards = [clone(base_tensor, i) for i in range(shard_count)]
    if issubclass(tensor_clazz, ReplicatedTensor):
        return ReplicatedTensor(ts=shards)

    if issubclass(tensor_clazz, SplitPrimitiveTensor):
        return SplitPrimitiveTensor(ts=shards, shard_dim=1)

    if issubclass(tensor_clazz, UnreducedTensor):
        return UnreducedTensor(ts=shards)

    raise TypeError(f"Unsupported tensor class {tensor_clazz}. ")


@dataclass
class OpTestConfig:
    """Configuration for testing op implementations.

    Attributes:
        op: The op from amdsharktank.ops (e.g., ops.scaled_dot_product_attention)
        reference_impl: Direct function reference to the reference implementation
        test_impls: List of implementations to test, or "all" to auto-discover all
        skip_impls: List of implementations to skip when test_impls="all"
        args: List of arguments to pass to the op (tensors or None for optional args)
        kwargs: Additional keyword arguments to pass to the op
        comparison_fn: Function to compare outputs (ref_output, test_output) -> None
                      Should raise AssertionError if outputs don't match
        fail_on_not_implemented: If True, fail test when implementation returns NotImplemented. If False, skip.
        impl_arg_transformers: Dict mapping implementation functions to argument transformer functions.
                              Each transformer takes (args, kwargs) and returns (new_args, new_kwargs).
    """

    op: Callable
    reference_impl: Callable
    test_impls: Optional[Union[List[Callable], str]] = "all"
    skip_impls: Optional[List[Callable]] = None
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    atol: float = 1e-3
    rtol: float = 1e-3
    comparison_fn: Callable[
        [Any, Any], None
    ] = lambda ref, test, *, rtol, atol, **_: assert_tensor_close(
        test, ref, rtol=rtol, atol=atol
    )
    fail_on_not_implemented: bool = True
    impl_arg_transformers: Dict[
        Callable,
        Callable[[List[Any], Dict[str, Any]], Tuple[List[Any], Dict[str, Any]]],
    ] = field(default_factory=dict)


class OpComparisonTestBase(unittest.TestCase):
    """Base class for comparing op implementations."""

    def _get_override_type_spec(self, op, override_func):
        """Get the type spec for an override function."""
        for override in op._overrides:
            if override.target == override_func:
                return override.type_spec
        raise ValueError(f"Could not find type spec for {override_func.__name__}")

    def _create_simple_block_scaled_quantizer(dtype):
        """Create a simple block-scaled quantized tensor for testing."""

        class SimpleBlockScaledQuantizer:
            def __init__(self, dtype):
                self.dtype = dtype

            def quantize(self, tensor):
                # Create a simple block scaled layout with block_size=1 (per-element scaling)
                # This makes it equivalent to tensor scaling but uses BlockScaledLayout structure
                d = torch.ones_like(tensor, dtype=torch.float32)  # scale per element
                qs = tensor.to(self.dtype)  # quantized values
                m = torch.zeros_like(tensor, dtype=torch.float32)  # zero offset

                layout = BlockScaledLayout(shape=list(tensor.shape), d=d, qs=qs, m=m)
                return PlanarQuantizedTensor(shape=list(tensor.shape), layout=layout)

        return SimpleBlockScaledQuantizer(dtype)

    LAYOUT_TO_QUANTIZER = {
        QuantizedTensor: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        PlanarQuantizedTensor: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        TensorScaledLayout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        BlockScaledLayout: lambda dtype=None: DynamicFp4BlockQuantizer(block_size=32),
        BlockScaledFp4Layout: lambda dtype=None: DynamicFp4BlockQuantizer(
            block_size=32,
        ),
        # TODO: Still need suitable default quantizers for:
        # BlockScaledI4Layout, SuperBlockOffsetScaled_4_6_Layout
    }

    def cast_inputs_for_override(
        self,
        op: Callable,
        override_func: Callable,
        args: List[Any],
        kwargs: Dict[str, Any] = None,
        config: OpTestConfig = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Cast inputs to match override signature types.

        Args:
            op: The operation being tested
            override_func: The override function
            args: List of input values
            kwargs: Keyword arguments
            config: Test configuration

        Returns:
            Tuple of (args, kwargs) cast to appropriate types
        """
        if kwargs is None:
            kwargs = {}

        # Apply argument transformer if one exists for this implementation (before casting)
        if config and override_func in config.impl_arg_transformers:
            transformer = config.impl_arg_transformers[override_func]
            transformed_args, transformed_kwargs = transformer(args, kwargs)
            return transformed_args, transformed_kwargs

        type_spec = self._get_override_type_spec(op, override_func)

        cast_args = cast_to_type_spec(args, type_spec, self.LAYOUT_TO_QUANTIZER)
        return cast_args, kwargs

    def compare_outputs(
        self,
        reference_output: Any,
        test_output: Any,
        config: OpTestConfig,
        impl_name: str,
    ):
        """Compare two outputs using the configured comparison function.

        Args:
            reference_output: Reference output
            test_output: Test output
            config: Test configuration
            impl_name: Name of the implementation being tested
        """

        reference_output = unbox_tensor(reference_output)
        test_output = unbox_tensor(test_output)

        try:
            config.comparison_fn(
                reference_output, test_output, atol=config.atol, rtol=config.rtol
            )
        except AssertionError as e:
            ref_name = config.reference_impl.__name__
            raise AssertionError(
                f"Implementation '{impl_name}' failed comparison against reference '{ref_name}': {e}"
            )

    def compare_implementations(self, config: OpTestConfig):
        """Main comparison method that tests all implementations.

        Args:
            config: Test configuration
        """
        if config.skip_impls is not None and config.test_impls != "all":
            return ValueError(
                'Invalid config. skip_impls should be None when test_impls != "all"'
            )
        all_impls = get_all_implementations(config.op)

        if not config.reference_impl:
            self.fail("No reference implementation specified")

        ref_name = config.reference_impl.__name__

        ref_args, ref_kwargs = self.cast_inputs_for_override(
            config.op, config.reference_impl, config.args, config.kwargs, config
        )
        ref_output = config.reference_impl(*ref_args, **ref_kwargs)

        if ref_output is NotImplemented:
            self.fail(f"Reference implementation '{ref_name}' returned NotImplemented")

        if config.test_impls != "all":
            test_impls = {func.__name__: func for func in config.test_impls}
        else:
            # Test all discovered implementations except the reference
            # TODO: Add support for testing sharded implementations by creating
            # appropriate sharded tensor inputs with distribution context
            test_impls = {}
            for name, func in all_impls.items():
                if name == ref_name:
                    continue
                # Skip implementations specified in skip_impls
                if config.skip_impls and func in config.skip_impls:
                    continue
                # Skip sharded implementations for now
                type_spec = self._get_override_type_spec(config.op, func)
                from amdsharktank.types import (
                    SplitPrimitiveTensor,
                    ReplicatedTensor,
                    UnreducedTensor,
                )

                has_sharded = any(
                    _matches(t, SplitPrimitiveTensor)
                    or _matches(t, ReplicatedTensor)
                    or _matches(t, UnreducedTensor)
                    for t in type_spec
                    if t is not None
                )
                if has_sharded:
                    continue
                test_impls[name] = func

        for impl_name in sorted(test_impls.keys()):
            impl_func = test_impls[impl_name]

            with self.subTest(implementation=impl_name):
                impl_args, impl_kwargs = self.cast_inputs_for_override(
                    config.op, impl_func, config.args, config.kwargs, config
                )
                impl_output = impl_func(*impl_args, **impl_kwargs)

                if impl_output is NotImplemented:
                    if config.fail_on_not_implemented:
                        self.fail(
                            f"Implementation '{impl_name}' returned NotImplemented"
                        )
                    else:
                        continue
                self.compare_outputs(ref_output, impl_output, config, impl_name)
