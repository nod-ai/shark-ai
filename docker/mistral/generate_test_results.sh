#!/bin/bash

# $1: test output file (raw)
# $2: iree benchmarks markdown file
# $3: cli benchmarks markdown file

test_output="$1"
iree_file="$2"
cli_file="$3"

current_time=$(date -u '+%Y-%m-%d %H:%M:%S')

# 1. IREE Benchmarks Table
mkdir -p "$(dirname "$iree_file")"
if [ ! -f "$iree_file" ]; then
    cat << EOF > "$iree_file"
## IREE Benchmarks

| Date (UTC)           | Benchmark                              | Time    | CPU     | Iterations | UserCounters                |
|----------------------|----------------------------------------|---------|---------|------------|-----------------------------|
EOF
fi

# Extract benchmark lines and append as rows
grep -E '^BM_' "$test_output" | while read -r line; do
    bench=$(echo "$line" | awk '{print $1}')
    time=$(echo "$line" | awk '{print $2 " " $3}')
    cpu=$(echo "$line" | awk '{print $4 " " $5}')
    iterations=$(echo "$line" | awk '{print $6}')
    usercounters=$(echo "$line" | cut -d' ' -f7-)
    echo "| $current_time | $bench | $time | $cpu | $iterations | $usercounters |" >> "$iree_file"
done

# 2. CLI Benchmarks Table
mkdir -p "$(dirname "$cli_file")"
if [ ! -f "$cli_file" ]; then
    cat << EOF > "$cli_file"
## CLI Benchmarks

| Date (UTC)           | Requests/sec | Latency av | min | max | median | sd |
|----------------------|--------------|------------|-----|-----|--------|----|
EOF
fi

# Extract requests per second
reqs=$(grep -m1 "Requests per second:" "$test_output" | awk '{print $4}')

# Extract latencies
lat_line=$(grep -m1 "Latencies:" "$test_output")
lat_av=$(echo "$lat_line" | sed -n 's/.*av: \([0-9.]*\),.*/\1/p')
lat_min=$(echo "$lat_line" | sed -n 's/.*min: \([0-9.]*\),.*/\1/p')
lat_max=$(echo "$lat_line" | sed -n 's/.*max: \([0-9.]*\),.*/\1/p')
lat_median=$(echo "$lat_line" | sed -n 's/.*median: \([0-9.]*\),.*/\1/p')
lat_sd=$(echo "$lat_line" | sed -n 's/.*sd: \([0-9.]*\)$/\1/p')

if [ -n "$reqs" ] && [ -n "$lat_line" ]; then
    echo "| $current_time | $reqs | $lat_av | $lat_min | $lat_max | $lat_median | $lat_sd |" >> "$cli_file"
fi

echo "Results appended to $iree_file and $cli_file"
