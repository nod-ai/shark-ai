# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
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

from sys import platform
from datasets import load_dataset

from sharktank.types import *
from sharktank.types.pipelining import pipeline_parallelize_theta
from .math import cosine_similarity

# TODO: ci-sharktank-nightly should run all nightly CIs and ci-sharktank/test-mi300x should run all pre-submits
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
is_deepseek = pytest.mark.skipif(
    'config.getoption("--deepseek-v3-model-path") is None',
    reason="Run deepseek tests if --deepseek-v3-model-path is passed",
)
is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")
is_cpu_condition = (
    "exec('from sharktank.utils.testing import is_iree_hal_target_device_cpu') or "
    "is_iree_hal_target_device_cpu(config.getoption('iree_hal_target_device'))"
)
is_not_cpu_condition = (
    "exec('from sharktank.utils.testing import is_iree_hal_target_device_cpu') or "
    "not is_iree_hal_target_device_cpu(config.getoption('iree_hal_target_device'))"
)
is_hip_condition = "config.getoption('iree_hal_target_device') == 'hip'"
is_cpu = pytest.mark.skipif(is_not_cpu_condition)
is_cpu_win = pytest.mark.skipif(is_cpu_condition and platform == "win32")


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
        """
        # Note: Here to prevent circular imports
        from sharktank.models.llm.llm import PagedLlmModelV1
        from sharktank.utils.evaluate import pad_tokens
        from sharktank.utils.load_llm import TorchGenerator
        from sharktank.utils.export_artifacts import ExportArtifacts

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
            from sharktank.types.sharding import shard_theta

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
            use_attention_mask=True,
        )

        # Note: Must be after saving the dataset and creating the exporter but before moving theta to the provided device.
        block_to_pipeline, pipeline_to_devices = pipeline_parallelize_theta(
            theta, self.config.pipeline_parallelism_size
        )
        self.config.block_to_pipeline_map = block_to_pipeline
        self.config.pipeline_to_device_map = pipeline_to_devices

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
            token_ids=prefill_token_ids, seq_lens=prefill_seq_lens, dump_path=work_dir
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

    If the env var SHARKTANK_TEST_ASSETS_DIR is set then directories will be
    created under there, named by `identifier`. If the `identifier` subdirectory
    exists, it will be deleted first.

    This is useful for getting updated goldens and such.
    """
    explicit_dir = os.getenv("SHARKTANK_TEST_ASSETS_DIR", None)
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
    elements_equal = elements_equal or eq
    for i, (v1, v2) in enumerate(zip(iterable1, iterable2, strict=True)):
        assert elements_equal(
            v1, v2
        ), f"Iterables not equal at index {i} for elements {v1} and {v2}"


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    max_outliers_fraction: Optional[float] = None,
    inlier_atol: Optional[float] = None,
):
    if (max_outliers_fraction is None and inlier_atol is not None) or (
        max_outliers_fraction is not None and inlier_atol is None
    ):
        raise ValueError(
            "max_outliers_fraction and inlier_atol must be provided or not together."
        )

    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=0,
        )

        if inlier_atol is not None:
            outliers = (actual - expected).abs() > inlier_atol
            outliers_fraction = outliers.count_nonzero() / outliers.numel()
            if outliers_fraction > max_outliers_fraction:
                raise AssertionError(
                    f"The fraction of outliers {outliers_fraction:%} is above the allowed "
                    f"{max_outliers_fraction:%}. Inlier atol={inlier_atol}."
                )
    except AssertionError as ex:
        diff = actual - expected
        std, mean = torch.std_mean(diff)
        msg = (
            "Difference (actual - expected):\n"
            f"mean = {mean}\n"
            f"median = {diff.median()}\n"
            f"std dev = {std}\n"
            f"min = {diff.min()}\n"
            f"max = {diff.max()}\n"
        )
        raise AssertionError(msg) from ex


def assert_cosine_similarity_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
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
        atol=atol,
        max_outliers_fraction=max_outliers_fraction,
        inlier_atol=inlier_atol,
    )


def assert_text_encoder_state_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
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
    actual: torch.Tensor,
    expected: torch.Tensor,
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
    actual_probabilities = actual.log_softmax(dim=2, dtype=torch.float32)
    expected_probabilities = expected.log_softmax(dim=2, dtype=torch.float32)

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss = kl_loss(input=actual_probabilities, target=expected_probabilities)

    assert torch.all(
        loss.abs() <= atol
    ), f"KL divergence loss {loss} is greater than the allowed tolerance {atol}."


SHARKTANK_TEST_SKIP_ENV_VAR = "SHARKTANK_TEST_SKIP"


def skip(*decorator_args, **decorator_kwargs):
    """Decorator to skip a test when SHARKTANK_TEST_SKIP env var is not set or != 0"""

    def decorator(test_item: Callable):
        if SHARKTANK_TEST_SKIP_ENV_VAR not in os.environ:
            should_skip = True
        else:
            should_skip = os.environ[SHARKTANK_TEST_SKIP_ENV_VAR] != "0"

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


def xfail(
    condition: bool | None = None,
    *,
    match: str | None = None,
    **kwargs,
):
    """xfail a test with support for regex matching against the error message.

    This wraps the pytest.mark.xfail decorator into a new decorator.
    pytest.mark.xfail does not support matching on the error message, but sometimes we
    need to be more precise on why we expect a failure.
    One example is when specifying what compiler error is expected. Just the exception
    type is not enough.

    ```
    @xfail(raises=MyError, strict=True, match="my message")
    @test_something():
        raise MyError("my message")
    ```

    *args and **kwargs are passthrough arguments for pytest.mark.xfail.
    """

    def decorator(test_fn: Callable):
        if condition is not None:
            kwargs.update(condition=condition)

        @pytest.mark.xfail(**kwargs)
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            try:
                return test_fn(*args, **kwargs)
            except Exception as ex:
                if (
                    not _eval_condition(condition)
                    or match is None
                    or re.search(match, str(ex))
                ):
                    raise ex
                else:
                    raise pytest.fail(
                        f'Failed to match error "{ex}" against expected match "{match}"'
                    ) from ex

        return wrapper

    return decorator


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
