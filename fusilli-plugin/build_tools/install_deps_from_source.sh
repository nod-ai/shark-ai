#!/bin/bash
# Install hipdnn_sdk dependencies from source
# These dependencies need CMake config files which aren't provided by apt packages

set -e

echo "Installing dependencies (using CMake default install location)..."

# Create temporary build directory
BUILD_DIR="/tmp/fusilli-plugin-deps-build-$$"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Number of parallel jobs
JOBS="${JOBS:-$(nproc)}"

echo "Using $JOBS parallel jobs for building"

# Install flatbuffers
echo "========================================="
echo "Building flatbuffers..."
echo "========================================="
git clone --depth 1 --branch v25.9.23 https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -S. -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DFLATBUFFERS_BUILD_FLATHASH=OFF \
    -DFLATBUFFERS_BUILD_FLATC=ON
cmake --build build -j$JOBS
cmake --install build
cd ..
echo "flatbuffers installed!"

# Install spdlog
echo "========================================="
echo "Building spdlog..."
echo "========================================="
git clone --depth 1 --branch v1.16.0 https://github.com/gabime/spdlog.git
cd spdlog
cmake -S. -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPDLOG_BUILD_EXAMPLE=OFF \
    -DSPDLOG_BUILD_TESTS=OFF
cmake --build build -j$JOBS
cmake --install build
cd ..
echo "spdlog installed!"

# Install nlohmann_json
echo "========================================="
echo "Building nlohmann_json..."
echo "========================================="
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
cd json
cmake -S. -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DJSON_BuildTests=OFF
cmake --build build -j$JOBS
cmake --install build
cd ..
echo "nlohmann_json installed!"

# Clean up
echo "========================================="
echo "Cleaning up..."
echo "========================================="
cd /
rm -rf "$BUILD_DIR"

echo "========================================="
echo "All dependencies installed successfully!"
echo "========================================="