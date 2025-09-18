import os
import tempfile

import lit.formats
from lit.llvm import llvm_config

config.name = "fusilli"

config.test_format = lit.formats.ShTest()

config.suffixes = [".cpp"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Without tempfile lit writes `.lit_test_times.txt` and an `Output` folder into
# the source tree.
config.test_exec_root = os.path.join(tempfile.gettempdir(), "lit")

# CMake provides the path of the executable who's output is being lit tested
# through a generator expression.
test_exe = lit_config.params.get("TEST_EXE")
if test_exe:
    config.substitutions.append(("%test_exe", test_exe))
