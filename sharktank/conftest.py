# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pytest
import re

from dataclasses import dataclass
from pytest import FixtureRequest
from typing import Any, Generator, Optional

from amdsharktank.utils.testing import IreeFlags

# Tests under each top-level directory will get a mark.
TLD_MARKS = {
    "tests": "unit",
    "integration": "integration",
}


def pytest_collection_modifyitems(items, config):
    # Add marks to all tests based on their top-level directory component.
    root_path = Path(__file__).parent
    for item in items:
        item_path = Path(item.path)
        rel_path = item_path.relative_to(root_path)
        tld = rel_path.parts[0]
        mark = TLD_MARKS.get(tld)
        if mark:
            item.add_marker(mark)


def pytest_addoption(parser):
    parser.addoption(
        "--mlir",
        type=Path,
        default=None,
        help="Path to exported MLIR program. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--module",
        type=Path,
        default=None,
        help="Path to exported IREE module. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--parameters",
        type=Path,
        default=None,
        help="Exported model parameters. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--prefix",
        type=str,
        default=None,
        help=(
            "Path prefix for test artifacts. "
            "Other arguments may override this for specific values."
        ),
    )
    parser.addoption(
        "--caching",
        action="store_true",
        default=False,
        help="Load cached results if present instead of recomputing.",
    )
    parser.addoption(
        "--device",
        type=str,
        action="store",
        default="cpu",
        help="List a torch device, (e.g., 'cuda:0')",
    )
    parser.addoption(
        "--run-quick-test",
        action="store_true",
        dest="run-quick-test",
        default=False,
        help="Enable all quick tests",
    )
    parser.addoption(
        "--run-nightly-test",
        action="store_true",
        dest="run-nightly-test",
        default=False,
        help="Enable all nightly tests",
    )
    parser.addoption(
        "--run-sharded-test",
        action="store_true",
        dest="run-sharded-test",
        default=False,
        help="Enable all sharded tests",
    )
    parser.addoption(
        "--with-clip-data",
        action="store_true",
        default=False,
        help=(
            "Enable tests that use CLIP data like models that is not a part of the source "
            "code. The user is expected to provide the data"
        ),
    )
    parser.addoption(
        "--with-flux-data",
        action="store_true",
        default=False,
        help=(
            "Enable tests that use Flux data like models that is not a part of the source "
            "code. The user is expected to provide the data"
        ),
    )
    parser.addoption(
        "--with-t5-data",
        action="store_true",
        default=False,
        help=(
            "Enable tests that use T5 data like models that is not a part of the source "
            "code. The user is expected to provide the data"
        ),
    )
    parser.addoption(
        "--with-vae-data",
        action="store_true",
        default=False,
        help=(
            "Enable tests that use vae data such as models not part of the source code."
        ),
    )
    parser.addoption(
        "--with-quark-data",
        action="store_true",
        default=False,
        help=(
            "Enable tests that use vae data such as models not part of the source code."
        ),
    )

    parser.addoption(
        "--test-data-dir",
        type=Path,
        action="store",
        help="Path to the directory containing external data for all tests. E.g. model weights. The subdirectory structure must comply with what is expected by the various tests.",
    )
    # TODO: Remove all hardcoded paths in CI tests
    parser.addoption(
        "--llama3-8b-tokenizer-path",
        type=Path,
        action="store",
        help="Llama3.1 8b tokenizer path",
    )
    parser.addoption(
        "--llama3-8b-f16-model-path",
        type=Path,
        action="store",
        help="Llama3.1 8b model path",
    )
    parser.addoption(
        "--llama3-8b-f8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 8b f8 model path",
    )
    parser.addoption(
        "--llama3-8b-f8-attnf8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 8b f8 attnf8 model path",
    )
    parser.addoption(
        "--llama3-70b-tokenizer-path",
        type=Path,
        action="store",
        help="Llama3.1 70b tokenizer path",
    )
    parser.addoption(
        "--llama3-70b-f16-model-path",
        type=Path,
        action="store",
        help="Llama3.1 70b model path",
    )
    parser.addoption(
        "--llama3-70b-f8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 70b f8 model path",
    )

    parser.addoption(
        "--baseline-perplexity-scores",
        type=Path,
        action="store",
        default="amdsharktank/tests/evaluate/baseline_perplexity_scores.json",
        help="Llama3.1 8B & 405B model baseline perplexity scores",
    )
    parser.addoption(
        "--iree-device",
        type=str,
        nargs="+",
        action="store",
        default="local-task",
        help="List an IREE device from 'iree-run-module --list_devices'",
    )
    parser.addoption(
        "--iree-hip-target",
        action="store",
        help="Specify the iree-hip target version (e.g., gfx942)",
    )
    parser.addoption(
        "--iree-hal-target-device",
        action="store",
        default="local",
        help="Specify the iree-hal target device (e.g., hip)",
    )
    parser.addoption(
        "--iree-hal-local-target-device-backends",
        type=list[str],
        nargs="+",
        action="store",
        default=["llvm-cpu"],
        help="Default target backends for local device executable compilation",
    )

    parser.addoption(
        "--tensor-parallelism-size",
        action="store",
        type=int,
        default=1,
        help="Number of devices for tensor parallel sharding",
    )
    parser.addoption(
        "--bs",
        action="store",
        type=int,
        default=4,
        help="Batch size for mlir export",
    )


def set_fixture(request: FixtureRequest, name: str, value: Any):
    if request.cls is None:
        return value
    else:
        setattr(request.cls, name, value)
        return value


def set_fixture_from_cli_option(
    request: FixtureRequest,
    cli_option_name: str,
    class_attribute_name: Optional[str] = None,
) -> Optional[Any]:
    value = request.config.getoption(cli_option_name)
    if class_attribute_name is None:
        class_attribute_name = cli_option_name
    return set_fixture(request, class_attribute_name, value)


@pytest.fixture(scope="class")
def mlir_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "mlir", "mlir_path")


@pytest.fixture(scope="class")
def module_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "module", "module_path")


@pytest.fixture(scope="class")
def parameters_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "parameters", "parameters_path")


@pytest.fixture(scope="class")
def path_prefix(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(request, "prefix", "path_prefix")


@pytest.fixture(scope="class")
def caching(request: FixtureRequest) -> Optional[bool]:
    return set_fixture_from_cli_option(request, "caching")


@pytest.fixture(scope="class")
def device(request: FixtureRequest) -> str:
    return set_fixture_from_cli_option(request, "device")


@pytest.fixture(scope="class")
def tensor_parallelism_size(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(
        request, "tensor_parallelism_size", "tensor_parallelism_size"
    )


@pytest.fixture(scope="class")
def baseline_perplexity_scores(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(
        request, "baseline_perplexity_scores", "baseline_perplexity_scores"
    )


@pytest.fixture(scope="class")
def batch_size(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(request, "bs", "batch_size")


@pytest.fixture(scope="class")
def model_artifacts(request: FixtureRequest) -> dict[str, str]:
    model_path = {}
    model_path["llama3_8b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-tokenizer-path", "llama3_8b_tokenizer"
    )
    model_path["llama3_8b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f16-model-path", "llama3_8b_f16_model"
    )
    model_path["llama3_8b_f8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f8-model-path", "llama3_8b_f8_model"
    )
    model_path["llama3_8b_f8_attnf8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f8-attnf8-model-path", "llama3_8b_f8_attnf8_model"
    )
    model_path["llama3_70b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-70b-tokenizer-path", "llama3_70b_tokenizer"
    )
    model_path["llama3_70b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-70b-f16-model-path", "llama3_70b_f16_model"
    )
    model_path["llama3_70b_f8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-70b-f8-model-path", "llama3_70b_f8_model"
    )
    return model_path


@pytest.fixture(scope="class")
def test_data_dir(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(
        request,
        "--test-data-dir",
        "test_data_dir",
    )


@pytest.fixture(scope="class")
def iree_flags(request: FixtureRequest) -> IreeFlags:
    iree_device = request.config.getoption("iree_device")
    if not isinstance(iree_device, str) and len(iree_device) == 1:
        iree_device = iree_device[0]
    set_fixture(request, "iree_device", iree_device)
    iree_hip_target = set_fixture_from_cli_option(
        request, "--iree-hip-target", "iree_hip_target"
    )
    iree_hal_target_device = set_fixture_from_cli_option(
        request, "--iree-hal-target-device", "iree_hal_target_device"
    )
    iree_hal_local_target_device_backends = set_fixture_from_cli_option(
        request,
        "--iree-hal-local-target-device-backends",
        "iree_hal_local_target_device_backends",
    )
    return IreeFlags(
        iree_device=iree_device,
        iree_hip_target=iree_hip_target,
        iree_hal_target_device=iree_hal_target_device,
        iree_hal_local_target_device_backends=iree_hal_local_target_device_backends,
    )


# The following three functions allow us to add a "XFail Reason" column to the html reports for each test
@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_header(cells):
    cells.insert(2, "<th>XFail Reason</th>")


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_row(report, cells):
    if hasattr(report, "wasxfail"):
        cells.insert(2, f"<td>{report.wasxfail}</td>")
    else:
        cells.insert(2, f"<td></td>")


def evaluate_mark_conditions(item: pytest.Item, mark: pytest.Mark) -> bool:
    # TODO: avoid non-API imports.
    from _pytest.skipping import evaluate_condition

    if "condition" not in mark.kwargs:
        conditions = mark.args
    else:
        conditions = (mark.kwargs["condition"],)

    if not conditions:
        return True

    return any(evaluate_condition(item, mark, condition)[0] for condition in conditions)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield

    if (
        report.when == "call"
        and hasattr(item, "wasxfail")
        and not hasattr(report, "wasxfail")
    ):
        report.wasxfail = item.wasxfail

    # Regex match against exception message.
    if call.when == "call":
        for xfail_mark in item.iter_markers(name="xfail"):
            if (
                item.config.option.runxfail
                or "match" not in xfail_mark.kwargs
                or (report.skipped and not hasattr(report, "wasxfail"))
            ):
                continue

            match = xfail_mark.kwargs["match"]
            if call.excinfo is not None:
                # TODO: avoid non-API imports.
                # The problem is that we want to skip matching on an explicitly xfailed
                # test from within the test.
                # Like that we are consistent with how pytest matches against the
                # exception type. Explicit xfails are skipped.
                from _pytest.outcomes import XFailed

                if isinstance(
                    call.excinfo.value, XFailed
                ) or not evaluate_mark_conditions(item, xfail_mark):
                    continue

                if not re.search(match, str(call.excinfo.value)):
                    report.outcome = "failed"
                    break

    return report


@pytest.fixture(scope="function")
def deterministic_random_seed():
    """Seed all RNGs (torch, numpy, builtin) with never-changing seed.
    At fixture teardown recover the original RNG states."""
    import torch
    import numpy as np
    import random
    from amdsharktank.utils.random import fork_numpy_singleton_rng, fork_builtin_rng

    with torch.random.fork_rng(), fork_numpy_singleton_rng(), fork_builtin_rng():
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        yield
