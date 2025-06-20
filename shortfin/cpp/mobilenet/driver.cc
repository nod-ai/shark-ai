// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <fstream>

#include "mobilenet.h"

static std::vector<float> readImageBinary(std::filesystem::path path) {
  std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
  if (!in.is_open()) {
    std::cerr << "Could not open file " << path << "\n";
    exit(1);
  }
  std::vector<float> raw_bin_data;
  float val;
  while (in.read(reinterpret_cast<char *>(&val), sizeof(val))) {
    raw_bin_data.push_back(static_cast<float>(val));
  }
  return raw_bin_data;
}

int main(int argc, const char **argv) {
  if (argc != 3) {
    std::cerr << "Invalid number of arguments.\nUsage: mnet </path/to/VMFB> "
                 "</path/to/raw/image/data/>"
              << std::endl;
    exit(1);
  }

  const char *vmfb_path = argv[1];
  const char *image_path = argv[2];

  std::vector<float> data = readImageBinary(fs::path{image_path});

  shortfin::cpp::Mobilenet mnet_server =
      shortfin::cpp::Mobilenet(fs::path{vmfb_path});

  shortfin::local::Worker::Options options(
      mnet_server.system().host_allocator(), "main-inference-worker");
  shortfin::local::Worker &runner = mnet_server.system().init_worker();
  // mnet_server.system().CreateWorker(std::move(options));

  auto coro = mnet_server.Run(data);
  coro.addDoneCallback([&]() { runner.Kill(); });

  runner.CallThreadsafe([&]() { coro.resume(); });
  runner.RunOnCurrentThread();

  mnet_server.Shutdown();
}
