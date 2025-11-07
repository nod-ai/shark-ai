#!/bin/bash
set -euo pipefail
set -x

# Arguments from CMake
RUN_BENCHMARK="$1"
DRIVER="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_COMMANDS="${SCRIPT_DIR}/test_commands.txt"
OUTPUT_CSV=$(mktemp)
python3 "${RUN_BENCHMARK}" \
  --commands-file "${TEST_COMMANDS}" \
  --csv "${OUTPUT_CSV}" \
  --driver "${DRIVER}" \
  --verbose
if [ ! -f "${OUTPUT_CSV}" ]; then
  echo "ERROR: Output CSV not created"
  exit 1
fi
# Count number of rows
NUM_ROWS=$(tail -n +2 "${OUTPUT_CSV}" | wc -l)
EXPECTED_ROWS=$(grep -c . "${TEST_COMMANDS}")
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

echo "PASSED: batch_profile test"
rm -f "${OUTPUT_CSV}"
