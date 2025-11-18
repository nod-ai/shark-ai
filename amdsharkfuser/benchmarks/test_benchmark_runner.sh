#!/bin/bash
set -euo pipefail
set -x

# Arguments from CMake
BENCHMARK_RUNNER="$1"
BENCHMARK_DRIVER="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_COMMANDS="${SCRIPT_DIR}/test_commands.txt"
OUTPUT_CSV=$(mktemp)
python3 "${BENCHMARK_RUNNER}" \
  --commands-file "${TEST_COMMANDS}" \
  --csv "${OUTPUT_CSV}" \
  --driver "${BENCHMARK_DRIVER}" \
  --verbose
if [ ! -f "${OUTPUT_CSV}" ]; then
  echo "ERROR: Output CSV not created"
  exit 1
fi
# Count number of rows
NUM_ROWS=$(tail -n +2 "${OUTPUT_CSV}" | wc -l)
# Count non-empty, non-comment lines (matching Python script behavior)
EXPECTED_ROWS=$(grep -Ev '^\s*#|^\s*$' "${TEST_COMMANDS}" | wc -l)
if [ "${NUM_ROWS}" -ne "${EXPECTED_ROWS}" ]; then
  echo "ERROR: Expected ${EXPECTED_ROWS} rows, got ${NUM_ROWS}"
  exit 1
fi
# Using --iter 10, check column exists and has value 10
if ! grep -q "iter" "${OUTPUT_CSV}"; then
  echo "ERROR: 'iter' column not found in CSV"
  exit 1
fi
# Check that dispatch_count column exists
if ! grep -q "dispatch_count" "${OUTPUT_CSV}"; then
  echo "ERROR: 'dispatch_count' column not found in CSV"
  exit 1
fi
# Verify at least one row has iter=10
if ! tail -n +2 "${OUTPUT_CSV}" | cut -d',' -f6 | grep -q "10"; then
  echo "ERROR: Expected iter=10 not found"
  exit 1
fi

echo "PASSED: fusilli_benchmark_runner_tests"
rm -f "${OUTPUT_CSV}"
