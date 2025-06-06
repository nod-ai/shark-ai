#include "utils.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <vector>

using float16_t = uint16_t;

#define OUTPUT_TY uint32_t
#define INPUT_TY float16_t

constexpr uint32_t recordRuns = 100u;
constexpr int ARGMAX_LABEL = 7; // Will still be top-1 here
constexpr int k = 8;

template <typename DataT>
static inline void fillValues(DataT *mat, uint32_t m, uint32_t n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j++) {
      // Fill top-K largest values at known locations
      mat[i * n + j] = (j >= ARGMAX_LABEL && j < ARGMAX_LABEL + k)
                           ? static_cast<DataT>(250.0 - (j - ARGMAX_LABEL))
                           : static_cast<DataT>(0.0);
    }
  }
}

template <typename DataT>
static inline void fillIndices(DataT *mat, uint32_t m, uint32_t n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j++) {
      // Fill top-K largest values at known locations
      mat[i * n + j] = j;
    }
  }
}

#define IREE_HAL_ROCM_MAX_KERNEL_ARG 128

std::vector<char> readFileIntoVector(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return std::vector<char>();
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();
  return buffer;
}

void benchmark_module(size_t reductionSize) {
  int batchSize = 1;

  std::vector<INPUT_TY> inputValues(batchSize * reductionSize);
  std::vector<OUTPUT_TY> inputIndices(batchSize * reductionSize);
  std::vector<OUTPUT_TY> outputIndices(batchSize * k);
  std::vector<INPUT_TY> outputValues(batchSize * k);

  fillValues(inputValues.data(), batchSize, reductionSize, k);
  fillIndices(inputIndices.data(), batchSize, reductionSize, k);

  std::cout << "Initializing device data..." << std::endl;
  INPUT_TY *d_input;
  OUTPUT_TY *d_indices;
  OUTPUT_TY *d_outputIndices;
  INPUT_TY *d_outputValues;

  size_t bytesInput = inputValues.size() * sizeof(INPUT_TY);
  size_t bytesIdx = inputIndices.size() * sizeof(OUTPUT_TY);
  size_t bytesOutIdx = outputIndices.size() * sizeof(OUTPUT_TY);
  size_t bytesOutVal = outputValues.size() * sizeof(INPUT_TY);

  CHECK_HIP_ERROR(hipMalloc(&d_input, bytesInput));
  CHECK_HIP_ERROR(hipMalloc(&d_indices, bytesIdx));
  CHECK_HIP_ERROR(hipMalloc(&d_outputIndices, bytesOutIdx));
  CHECK_HIP_ERROR(hipMalloc(&d_outputValues, bytesOutVal));

  CHECK_HIP_ERROR(hipMemcpy(d_input, inputValues.data(), bytesInput,
                            hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_indices, inputIndices.data(), bytesIdx,
                            hipMemcpyHostToDevice));

  hipModule_t module;
  hipFunction_t kernel;
  std::vector<char> hsacoVec =
      readFileIntoVector("compiled_kernels/topk_fp16_ukernel.c.hsaco");
  if (hipModuleLoadDataEx(&module, hsacoVec.data(), 0, nullptr, nullptr) !=
      hipSuccess) {
    std::cerr << "Failed to load module!" << std::endl;
    return;
  }
  if (hipModuleGetFunction(&kernel, module, "topk_F16I32") != hipSuccess) {
    std::cerr << "Failed to get function!" << std::endl;
    return;
  }

  // Prepare kernel arguments
  size_t block_dimx = 32;
  size_t block_dimy = 1;
  int gridX = batchSize;
  int gridY = 1;
  void **kernelParam =
      (void **)malloc(IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(void *));
  hipDeviceptr_t *device_ptrs = (hipDeviceptr_t *)malloc(
      IREE_HAL_ROCM_MAX_KERNEL_ARG * sizeof(hipDeviceptr_t));
  for (size_t i = 0; i < IREE_HAL_ROCM_MAX_KERNEL_ARG; i++) {
    kernelParam[i] = &device_ptrs[i];
  }

  *((hipDeviceptr_t *)kernelParam[0]) = d_input;
  *((hipDeviceptr_t *)kernelParam[1]) = d_indices;
  *((hipDeviceptr_t *)kernelParam[2]) = d_outputValues;
  *((hipDeviceptr_t *)kernelParam[3]) = d_outputIndices;
  *((uint32_t *)kernelParam[4]) = static_cast<uint32_t>(reductionSize);
  *((uint32_t *)kernelParam[5]) = static_cast<uint32_t>(k);

  // Launch
  std::cout << "Launching Topk kernel..." << std::endl;
  hipEvent_t startEvent, stopEvent;
  CHECK_HIP_ERROR(hipEventCreate(&startEvent));
  CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
  CHECK_HIP_ERROR(hipEventRecord(startEvent));

  for (uint32_t i = 0; i < recordRuns; ++i) {
    assert(hipModuleLaunchKernel(kernel, gridX, gridY, 1, block_dimx,
                                 block_dimy, 1, 0, nullptr, kernelParam,
                                 nullptr) == 0);
  }

  CHECK_HIP_ERROR(hipEventRecord(stopEvent));
  CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

  float elapsedTimeMs = 0.0f;
  CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
  CHECK_HIP_ERROR(hipEventDestroy(startEvent));
  CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

  CHECK_HIP_ERROR(hipMemcpy(outputIndices.data(), d_outputIndices, bytesOutIdx,
                            hipMemcpyDeviceToHost));
  CHECK_HIP_ERROR(hipMemcpy(outputValues.data(), d_outputValues, bytesOutVal,
                            hipMemcpyDeviceToHost));

  // Validate
  std::vector<OUTPUT_TY> expected;
  for (int i = 0; i < k; ++i)
    expected.push_back(ARGMAX_LABEL + i);
  std::sort(outputIndices.begin(), outputIndices.end());

  std::cout << "Top-K indices: ";
  for (int i = 0; i < k; ++i)
    std::cout << outputIndices[i] << " ";
  std::cout << "\n";

  if (!std::equal(outputIndices.begin(), outputIndices.end(),
                  expected.begin())) {
    std::cerr << "Validation failed! Expected topk indices: ";
    for (int i : expected)
      std::cout << i << " ";
    std::cout << std::endl;
    exit(1);
  }

  std::cout << "Top-K kernel validated successfully!" << std::endl;
  std::cout << "Average time per run: " << (elapsedTimeMs / recordRuns)
            << " ms/iter" << std::endl;

  CHECK_HIP_ERROR(hipFree(d_input));
  CHECK_HIP_ERROR(hipFree(d_outputIndices));
  CHECK_HIP_ERROR(hipFree(d_outputValues));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " reductionSize" << std::endl;
    // std::cout << "Usage: " << argv[1] << " K" << std::endl;
    return 1;
  }
  size_t reductionSize = atoi(argv[1]);
  // int k = atoi(argv[2]);
  benchmark_module(reductionSize);
  return 0;
}
