// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the Backend type and code to map from Backend to
// `iree-compile` flags.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_H
#define FUSILLI_BACKEND_H

#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {
// Where do we want the generated code to run?
enum class Backend {
  CPU,
  GFX942,
};

// The flags corresponding to each compile backend.
static const std::unordered_map<Backend, std::vector<std::string>>
    backendFlags = {
        {
            Backend::CPU,
            {
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
            },
        },
        {
            Backend::GFX942,
            {
                "--iree-hal-target-backends=rocm",
                "--iree-hip-target=gfx942",
            },
        },
};

} // namespace fusilli
#endif // FUSILLI_BACKEND_H
