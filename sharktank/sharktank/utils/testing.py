# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import contextlib
from pathlib import Path
import pytest
from os import PathLike
import os
import shutil
import tempfile
import unittest
from typing import Any, Callable
from operator import eq
from collections.abc import Iterable
import gc
import random
import torch

from datasets import load_dataset

from sharktank.types import *
from .math import cosine_similarity

# TODO: Remove once pre-submits and nightly tests are unified to single workflow.
def get_test_type():
    pre_submit = 'config.getoption("--run-quick-llama-test")'
    nightly = 'config.getoption("--run-nightly-llama-tests")'
    if pre_submit or nightly:
        return False
    else:
        return True


is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")

# TODO: ci-sharktank-nightly should run all nightly CIs requiring mi300x in a single workflow, dropping all test specific flags/workflows
is_nightly = pytest.mark.skipif(
    'not config.getoption("run-nightly-llama-tests")',
    reason="Run large tests if --run-nightly-llama-tests is passed",
)

# TODO: ci-sharktank/test-mi300x should run all pre-submits requiring mi300x in a single workflow, dropping all test specific flags/workflows
is_pre_submit_nightly = pytest.mark.skipif(
    get_test_type(),
    reason="Run large/quick tests if --run-quick-llama-test or --run-nightly-llama-tests is passed",
)
is_llama_8b = pytest.mark.skipif(
    'config.getoption("llama3_8b_f16_model_path") is None',
    reason="Run llama tests if --llama3-8b-f16-model-path is passed",
)
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


def is_iree_hal_target_device_cpu(v: str, /) -> bool:
    return v.startswith("local") or v == "llvm-cpu"


def get_iree_compiler_flags(o: Any, device_count: int = 1) -> list[str]:
    """Retrieve compiler flags driven by the test configuration."""
    res = []
    if device_count == 1:
        res += [f"--iree-hal-target-device={o.iree_hal_target_device}"]
    else:
        res += [
            f"--iree-hal-target-device={o.iree_hal_target_device}[{i}]"
            for i in range(device_count)
        ]
    if o.iree_hal_target_device.startswith("local"):
        res += [
            f"--iree-hal-local-target-device-backends={v}"
            for v in o.iree_hal_local_target_device_backends
        ]
        res += ["--iree-llvmcpu-target-cpu=host"]
    elif o.iree_hal_target_device.startswith("hip"):
        res += [f"--iree-hip-target={o.iree_hip_target}"]
    return res


# Range of torch.rand() is [0,1)
# Range of torch.rand() * 2 - 1 is [-1, 1), includes negative values
def make_rand_torch(shape: list[int], dtype: Optional[torch.dtype] = torch.float32):
    return (torch.rand(shape) * 2 - 1).to(dtype=dtype)


def make_random_mask(shape: tuple[int], dtype: Optional[torch.dtype] = None):
    mask = make_rand_torch(shape=shape, dtype=dtype)
    mask = (mask >= 0).to(dtype=dtype)
    return mask


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
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
):
    """Asserts that actual and reference safetensors files are within tolerances.

    actual_path and ref_path can be directories. In that case files with matching
    sub-paths will be compared."""
    from safetensors import safe_open
    import torch

    print(f"Asserting tensors close: actual={actual_path}, ref={ref_path}")

    not_close_list: list[tuple[str, str]] = []

    if os.path.isdir(ref_path):
        assert os.path.isdir(actual_path)
        actual_path = os.path.abspath(actual_path)
        ref_path = os.path.abspath(ref_path)
        ref_paths = [
            Path(root) / file
            for root, dirs, files in os.walk(ref_path)
            for file in files
        ]
        for ref_file_path in ref_paths:
            actual_file_path = Path(
                f"{actual_path}{str(ref_file_path).removeprefix(ref_path)}"
            )
            try:
                assert os.path.isfile(actual_file_path)
                assert_close_safetensors(
                    actual_file_path,
                    ref_file_path,
                    rtol=rtol,
                    atol=atol,
                    fail_fast=fail_fast,
                )
            except Exception as ex:
                if fail_fast:
                    raise
                not_close_list.append((actual_file_path, ref_file_path))
                print(ex)
        if len(not_close_list) > 0:
            print("Not close:")
            for actual, ref in not_close_list:
                print(f"{actual} != {ref}")
            assert False, "Tensors are not close."

    def print_stats(label, t):
        t = t.to(dtype=torch.float32)
        std, mean = torch.std_mean(t)
        print(
            f"    {label}: "
            f"MIN={torch.min(t)}, "
            f"MAX={torch.max(t)}, "
            f"MEAN={mean}, STD={std}"
        )

    with safe_open(actual_path, framework="pt") as actual_f, safe_open(
        ref_path, framework="pt"
    ) as ref_f:
        is_close = True
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
                torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)
            except Exception as ex:
                if fail_fast:
                    raise
                is_close = False
                print(ex)

        assert is_close, "Tensors are not close."


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
