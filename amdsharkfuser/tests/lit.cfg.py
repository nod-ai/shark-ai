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
# the source tree. Use a unique tempdir for each test to prevent interference
# when tests are run concurrently (using `ctest -j N`), as they all try to
# read/write to the same `.lit_test_times.txt` causing malformed text.
config.test_exec_root = tempfile.mkdtemp(prefix="lit-test-")

# Setting `FUSILLI_CACHE_DIR=/tmp` helps bypass file access issues on
# LIT tests that rely on dumping/reading intermediate compilation artifacts
# to/from disk.
config.environment["FUSILLI_CACHE_DIR"] = "/tmp"

# Configure CHECK prefix for backend specific tests
backend = lit_config.params.get("BACKEND")
if backend:
    config.substitutions.append(("%{BACKEND}", backend))

# CMake provides the path of the executable who's output is being lit tested
# through a generator expression.
test_exe = lit_config.params.get("TEST_EXE")
if test_exe:
    config.substitutions.append(("%{TEST_EXE}", test_exe))
