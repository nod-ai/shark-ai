#!/bin/bash

# $1: test output file (raw)
# $2: results markdown file

test_output="$1"
results_file="$2"

# Parse values
batch_size=$(grep -i "Batch size" "$test_output" | awk '{print $3}')
total_requests=$(grep -i "total requests" "$test_output" | awk '{print $3}')
errors=$(grep -i "errors" "$test_output" | awk '{print $2}')
throughput_req=$(grep -i "throughput (request / second)" "$test_output" | awk '{print $5}')
throughput_char=$(grep -i "throughput (character / second)" "$test_output" | awk '{print $5}')
avg_req_duration=$(grep -i "average request duration (s)" "$test_output" | awk '{print $5}')
p50_req_duration=$(grep -i "50%ile request duration (s)" "$test_output" | awk '{print $5}')

# Get current time in UTC
current_time=$(date -u '+%Y-%m-%d %H:%M:%S')

# Ensure the results directory exists
mkdir -p "$(dirname "$results_file")"

# If file doesn't exist, write header
if [ ! -f "$results_file" ]; then
    cat << EOF > "$results_file"
## Test Results

| Run Date (UTC)        | Batch size | Total requests | Errors | Throughput (req/s) | Throughput (char/s) | Avg req duration (s) | 50%ile req duration (s) |
|-----------------------|------------|---------------|--------|--------------------|---------------------|----------------------|-------------------------|
EOF
fi

# Append new row
echo "| $current_time | $batch_size | $total_requests | $errors | $throughput_req | $throughput_char | $avg_req_duration | $p50_req_duration |" >> "$results_file"

echo "Results appended to $results_file"
