# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

from pathlib import Path
import sharktank.tools.status


@pytest.mark.skipif(
    sys.platform == "win32", reason="The status tool is not a Windows tool."
)
def test_status(tmp_path: Path):
    out_file_path = tmp_path / "status.log"
    sharktank.tools.status.main(
        [
            f"--out-file={out_file_path}",
            "--period=0.5",
            "--",
            f"{sys.executable}",
            "-c",
            "import time; time.sleep(1)",
        ]
    )
    assert out_file_path.exists()
    with open(out_file_path, "r") as f:
        status_contents = f.read()
    assert status_contents.find("Pid:") != -1
