# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path
from typing import Callable

from boo_tuner.boo_tuner import (
    BooPathConfig,
    load_commands_from_file_or_args,
    insert_placeholder_input_file,
    build_compile_args,
)


@pytest.fixture
def tmp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary files.

    Returns a callable that takes (content: str, suffix: str) and returns a Path
    to a temporary file with that content and suffix. Cleanup is handled by pytest's
    tmp_path fixture.
    """
    counter = 0

    def _create(content: str, suffix: str = ".txt") -> Path:
        nonlocal counter
        counter += 1
        temp_file = tmp_path / f"test_file_{counter}{suffix}"
        temp_file.write_text(content)
        return temp_file

    return _create


def test_load_commands_no_file() -> None:
    # Test without tabs.
    miopen_op_args = ["arg1", "arg2", "arg3"]
    result = load_commands_from_file_or_args(None, miopen_op_args)
    assert result == [["arg1", "arg2", "arg3"]]

    # Test with tab-separated arguments.
    miopen_op_args = ["arg1\targ2", "arg3"]
    result = load_commands_from_file_or_args(None, miopen_op_args)
    assert result == [["arg1", "arg2", "arg3"]]


def test_load_commands_from_text_file(tmp_file: Callable[[str, str], Path]) -> None:
    content = "# Comment line\ncmd1 arg1 arg2\n\ncmd2 'quoted arg' arg3\n"
    file_path = tmp_file(content, ".txt")

    result = load_commands_from_file_or_args(str(file_path), [])

    assert len(result) == 2
    assert result[0] == ["cmd1", "arg1", "arg2"]
    assert result[1] == ["cmd2", "quoted arg", "arg3"]


def test_load_commands_from_tsv_file(tmp_file: Callable[[str, str], Path]) -> None:
    content = "# Header comment\ncmd1\targ1\targ2\n\ncmd2\targ3\targ4\n"
    file_path = tmp_file(content, ".tsv")

    result = load_commands_from_file_or_args(str(file_path), [])

    assert len(result) == 2
    assert result[0] == ["cmd1", "arg1", "arg2"]
    assert result[1] == ["cmd2", "arg3", "arg4"]


def test_load_commands_file_and_args_raises_error(
    tmp_file: Callable[[str, str], Path]
) -> None:
    content = "cmd1 arg1\n"
    file_path = tmp_file(content, ".txt")

    with pytest.raises(
        ValueError,
        match="Cannot specify both --commands-file and MIOpen operation arguments",
    ):
        load_commands_from_file_or_args(str(file_path), ["extra", "args"])


def test_insert_placeholder_input_file() -> None:
    # Test with no additional args.
    argv = ["boo-tuner"]
    result = insert_placeholder_input_file(argv)
    assert result == ["boo-tuner", "boo.mlir"]

    # Test with multiple flags.
    argv = ["boo-tuner", "--commands-file", "cmds.txt", "--devices", "hip://0"]
    result = insert_placeholder_input_file(argv)
    assert result == [
        "boo-tuner",
        "boo.mlir",
        "--commands-file",
        "cmds.txt",
        "--devices",
        "hip://0",
    ]


def test_build_compile_args() -> None:
    compile_command = (
        "iree-compile --iree-hal-target-backends=rocm --iree-hip-target=mi300x "
        "--iree-opt-level=O3 /path/to/input.mlir -o /path/to/output.vmfb"
    )
    benchmarks_dir = Path("/tmp/benchmarks")

    result = build_compile_args(compile_command, benchmarks_dir)

    # Check that result starts with iree-compile.
    assert result[0] == "iree-compile"

    # Check that original flags are preserved.
    assert "--iree-hal-target-backends=rocm" in result
    assert "--iree-hip-target=mi300x" in result
    assert "--iree-opt-level=O3" in result
    assert "/path/to/input.mlir" in result

    # Check that -o and its argument are removed from original command.
    assert result.count("-o") == 1
    assert "/path/to/output.vmfb" not in result

    # Check that tuner-specific flags are added.
    assert "--iree-config-add-tuner-attributes" in result
    assert "--iree-hal-dump-executable-benchmarks-to" in result
    assert str(benchmarks_dir) in result

    o_index = result.index("-o")
    assert result[o_index + 1] == "/dev/null"


def test_path_config() -> None:
    boo_path_config = BooPathConfig()
    benchmark_name = "gemm_dispatch_0"
    benchmark_path_config = boo_path_config.create_benchmark_path_config(benchmark_name)

    expected_base = boo_path_config.base_dir / benchmark_name
    assert benchmark_path_config.base_dir == expected_base
    assert benchmark_path_config.template_mlir == expected_base / "template.mlir"
    assert benchmark_path_config.candidates_dir == expected_base / "candidates"
    assert (
        benchmark_path_config.compiled_dir == expected_base / "candidates" / "compiled"
    )
    assert benchmark_path_config.specs_dir == expected_base / "candidates" / "specs"
