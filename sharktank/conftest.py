# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pytest
from pytest import FixtureRequest
from typing import Optional, Any


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
        help="List a torch device, (e.g., 'cuda:0')",
    )
    parser.addoption(
        "--run-quick-llama-test",
        action="store_true",
        dest="run-quick-llama-test",
        default=False,
        help="Run large llama tests if passed",
    )
    parser.addoption(
        "--run-nightly-llama-tests",
        action="store_true",
        dest="run-nightly-llama-tests",
        default=False,
        help="Enable all llama benchmarking tests",
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

    # TODO: Remove all hardcoded paths in CI tests
    parser.addoption(
        "--llama3-8b-tokenizer-path",
        type=Path,
        action="store",
        help="Llama3.1 8b tokenizer path, defaults to 30F CI system path",
    )
    parser.addoption(
        "--llama3-8b-f16-model-path",
        type=Path,
        action="store",
        help="Llama3.1 8b model path, defaults to 30F CI system path",
    )
    parser.addoption(
        "--llama3-8b-f16-tp2-model-path",
        type=Path,
        action="store",
        help="Llama3.1 8b tp2 model path, defaults to 30F CI system path",
    )
    parser.addoption(
        "--llama3-8b-f8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 8b f8 model path",
    )
    parser.addoption(
        "--llama3-405b-tokenizer-path",
        type=Path,
        action="store",
        help="Llama3.1 405b tokenizer path, defaults to 30F CI system path",
    )
    parser.addoption(
        "--llama3-405b-f16-model-path",
        type=Path,
        action="store",
        help="Llama3.1 405b model path, defaults to 30F CI system path",
    )
    parser.addoption(
        "--llama3-405b-f8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 405b f8 model path",
    )

    # To obtain a T5 GGUF file you can use llama.cpp's convert_hf_to_gguf.py.
    # https://github.com/ggerganov/llama.cpp/blob/9abe9eeae98b11fa93b82632b264126a010225ff/convert_hf_to_gguf.py
    # E.g.
    # git lfs install
    # git clone https://huggingface.co/google/t5-v1_1-small
    # convert_hf_to_gguf.py \
    #     --outfile t5-v1_1-small.gguf \
    #     --outtype=f32 \
    #     t5-v1_1-small
    parser.addoption(
        "--google-t5-v1-1-small-f32-model-path",
        type=Path,
        default="/data/t5/small/google__t5-v1_1-small_f32.gguf",
        help="Google T5 v1.1 small float32 model path",
    )
    parser.addoption(
        "--google-t5-v1-1-small-bf16-model-path",
        type=Path,
        default="/data/t5/small/google__t5-v1_1-small_bf16.gguf",
        help="Google T5 v1.1 small bfloat16 model path",
    )
    parser.addoption(
        "--google-t5-v1-1-xxl-f32-model-path",
        type=Path,
        default="/data/t5/xxl/google__t5-v1_1-xxl_f32.gguf",
        help="Google T5 v1.1 XXL float32 model path",
    )
    parser.addoption(
        "--google-t5-v1-1-xxl-bf16-model-path",
        type=Path,
        default="/data/t5/xxl/google__t5-v1_1-xxl_bf16.gguf",
        help="Google T5 v1.1 XXL bfloat16 model path",
    )

    parser.addoption(
        "--baseline-perplexity-scores",
        type=Path,
        action="store",
        default="sharktank/tests/evaluate/baseline_perplexity_scores.json",
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
def device(request: FixtureRequest) -> Optional[bool]:
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
def get_model_artifacts(request: FixtureRequest):
    model_path = {}
    model_path["llama3_8b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-tokenizer-path", "llama3_8b_tokenizer"
    )
    model_path["llama3_8b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f16-model-path", "llama3_8b_f16_model"
    )
    model_path["llama3_8b_f16_tp2_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f16-tp2-model-path", "llama3_8b_f16_tp2_model"
    )
    model_path["llama3_8b_f8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f8-model-path", "llama3_8b_f8_model"
    )
    model_path["llama3_405b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-tokenizer-path", "llama3_405b_tokenizer"
    )
    model_path["llama3_405b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-f16-model-path", "llama3_405b_f16_model"
    )
    model_path["llama3_405b_f8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-f8-model-path", "llama3_405b_f8_model"
    )
    model_path["google__t5_v1_1_small_f32_model_path"] = set_fixture_from_cli_option(
        request,
        "--google-t5-v1-1-small-f32-model-path",
        "google__t5_v1_1_small_f32_model",
    )
    model_path["google__t5_v1_1_small_bf16_model_path"] = set_fixture_from_cli_option(
        request,
        "--google-t5-v1-1-small-bf16-model-path",
        "google__t5_v1_1_small_bf16_model",
    )
    model_path["google__t5_v1_1_xxl_f32_model_path"] = set_fixture_from_cli_option(
        request,
        "--google-t5-v1-1-xxl-f32-model-path",
        "google__t5_v1_1_xxl_f32_model",
    )
    return model_path


@pytest.fixture(scope="class")
def get_iree_flags(request: FixtureRequest):
    model_path = {}
    iree_device = request.config.getoption("iree_device")
    if not isinstance(iree_device, str) and len(iree_device) == 1:
        iree_device = iree_device[0]
    set_fixture(request, "iree_device", iree_device)
    model_path["iree_hip_target"] = set_fixture_from_cli_option(
        request, "--iree-hip-target", "iree_hip_target"
    )
    model_path["iree_hal_target_device"] = set_fixture_from_cli_option(
        request, "--iree-hal-target-device", "iree_hal_target_device"
    )
    model_path["iree_hal_local_target_device_backends"] = set_fixture_from_cli_option(
        request,
        "--iree-hal-local-target-device-backends",
        "iree_hal_local_target_device_backends",
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and hasattr(item, "wasxfail"):
        report.wasxfail = item.wasxfail
