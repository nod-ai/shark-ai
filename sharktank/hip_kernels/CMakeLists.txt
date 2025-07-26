cmake_minimum_required(VERSION 3.13)
project(HIP_KERNELS LANGUAGES C CXX)


# Determine IREE_COMPILER_DIR from environment or Python
if(NOT DEFINED ENV{IREE_COMPILER_DIR} OR "$ENV{IREE_COMPILER_DIR}" STREQUAL "")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import iree.compiler, os; print(os.path.dirname(iree.compiler.__file__))"
    OUTPUT_VARIABLE IREE_COMPILER_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(IREE_COMPILER_DIR $ENV{IREE_COMPILER_DIR})
endif()

message(STATUS "IREE_COMPILER_DIR: ${IREE_COMPILER_DIR}")

# Set ROCm and IREE build paths
set(ROCM_PATH /opt/rocm CACHE STRING "Path to ROCM library")
set(TARGET_ARCH "gfx942" CACHE STRING "Target architecture for ROCM GPU")

if (NOT IS_DIRECTORY "${ROCM_PATH}/amdgcn")
  message(SEND_ERROR "amdgcn folder in ROCM path not found")
endif()

if (NOT IS_DIRECTORY "${IREE_COMPILER_DIR}/_mlir_libs/iree_platform_libs/rocm")
  message(SEND_ERROR "iree rocm platform libs not found")
endif()

function (find_required_program binary BINARY_VAR)
  unset (LOCAL_BINARY_VAR CACHE)
  find_program(LOCAL_BINARY_VAR ${binary})
  if (NOT LOCAL_BINARY_VAR)
    message(SEND_ERROR "Command not found ${binary}")
  endif()
  SET(${BINARY_VAR} ${LOCAL_BINARY_VAR} PARENT_SCOPE)
endfunction()

find_required_program(clang-19 CLANG_19)
find_required_program(lld-19 LLD_19)
find_required_program(llvm-link-19 LLVM_LINK_19)

# Find all kernel source files in the kernels subdirectory
file(GLOB KERNEL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.c")
file(GLOB SPEC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/specs/*.mlir")
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

# Paths to ROCm and IREE bitcode libraries
set(ROCM_BC
  ${ROCM_PATH}/amdgcn/bitcode/opencl.bc
  ${ROCM_PATH}/amdgcn/bitcode/hip.bc
  ${ROCM_PATH}/amdgcn/bitcode/oclc_isa_version_1100.bc
)
set(IREE_BC
  ${IREE_COMPILER_DIR}/_mlir_libs/iree_platform_libs/rocm/ockl.bc
  ${IREE_COMPILER_DIR}/_mlir_libs/iree_platform_libs/rocm/ocml.bc
)

set(HSACO_TARGETS "")

# Define the output directory for .hsaco files
set(CUSTOM_KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled_kernels")
file(MAKE_DIRECTORY ${CUSTOM_KERNEL_OUTPUT_DIR})

foreach(KERNEL_SRC ${KERNEL_SRCS})
  get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME)
  set(BC_FILE "${KERNEL_NAME}.bc")
  set(LINKED_BC_FILE "${KERNEL_NAME}.linked.bc")
  set(O_FILE "${KERNEL_NAME}.o")
  set(HSACO_FILE "${KERNEL_NAME}.hsaco")
  set(HSACO_OUTPUT_FILE "${CUSTOM_KERNEL_OUTPUT_DIR}/${KERNEL_NAME}.hsaco")

  # Step 1: Compile HIP kernel to LLVM IR
  add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CLANG_19}
      -x hip --offload-arch=${TARGET_ARCH} --offload-device-only -nogpulib
      -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -O3 -fvisibility=protected
      -emit-llvm -c ${KERNEL_SRC} -o ${BC_FILE}
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling ${KERNEL_NAME} to LLVM IR"
  )

  # Step 2: Link with ROCm/IREE bitcode
  add_custom_command(
    OUTPUT ${LINKED_BC_FILE}
    COMMAND ${LLVM_LINK_19}
      ${IREE_BC} ${ROCM_BC} ${BC_FILE} -o ${LINKED_BC_FILE}
    DEPENDS ${BC_FILE}
    COMMENT "Linking ${KERNEL_NAME} with ROCm/IREE bitcode"
  )

  # Step 3: Compile to AMDGPU object file
  add_custom_command(
    OUTPUT ${O_FILE}
    COMMAND ${CLANG_19}
      -target amdgcn-amd-amdhsa -mcpu=${TARGET_ARCH}
      -c ${LINKED_BC_FILE} -o ${O_FILE}
    DEPENDS ${LINKED_BC_FILE}
    COMMENT "Compiling ${KERNEL_NAME} linked bitcode to AMDGPU object"
  )

  # Step 4: Link to produce .hsaco
  add_custom_command(
    OUTPUT ${HSACO_FILE}
    COMMAND ${LLD_19}
      -flavor gnu -shared ${O_FILE} -o ${HSACO_FILE}
    DEPENDS ${O_FILE}
    COMMENT "Linking ${KERNEL_NAME} object to produce .hsaco"
  )

  # Step 5: Copy .hsaco to output directory
  add_custom_command(
    OUTPUT ${HSACO_OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${HSACO_FILE} ${HSACO_OUTPUT_FILE}
    DEPENDS ${HSACO_FILE}
    COMMENT "Copying ${HSACO_FILE} to ${HSACO_OUTPUT_FILE}"
  )

  # Collect all .hsaco output targets
  list(APPEND HSACO_TARGETS ${HSACO_OUTPUT_FILE})

  # Optional: Clean up intermediates
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${BC_FILE}
    ${LINKED_BC_FILE}
    ${O_FILE}
    ${HSACO_FILE}
  )
endforeach()


# Define the output directory for .hsaco files
set(CUSTOM_SPEC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/specs")
file(MAKE_DIRECTORY ${CUSTOM_SPEC_OUTPUT_DIR})

foreach(SPEC_SRC ${SPEC_SRCS})
  get_filename_component(SPEC_NAME ${SPEC_SRC} NAME)
  set(SPEC_FILE "${CUSTOM_SPEC_OUTPUT_DIR}/${SPEC_NAME}")

  add_custom_command(
    OUTPUT ${SPEC_FILE}
    COMMAND sed -e 's,{{HIP_ARCH}},${TARGET_ARCH},' -e 's,{{BUILD}},${CMAKE_CURRENT_BINARY_DIR},'  ${SPEC_SRC} > ${SPEC_FILE}
    DEPENDS ${SPEC_SRC}
    COMMENT "Filling template ${HSACO_FILE} to ${TARGET_ARCH}"
  )

  list(APPEND HSACO_TARGETS ${SPEC_FILE})

endforeach()

# Define the output directory for .hsaco files
set(TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests")
file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})
set(TEST_TARGETS "")

foreach(TEST_SRC ${TEST_SRCS})
  get_filename_component(TEST_NAME ${TEST_SRC} NAME)
  set(TEST_FILE "${TEST_OUTPUT_DIR}/${TEST_NAME}")

  add_custom_command(
    OUTPUT ${TEST_FILE}
    COMMAND hipcc ${TEST_SRC} -std=c++20 -o ${TEST_FILE}
    DEPENDS ${TEST_SRC}
    COMMENT "Compiling ${TEST_FILE} test"
  )

  list(APPEND TEST_TARGETS ${TEST_FILE})
endforeach()

# Add a custom target to build all .hsaco files in the output directory
add_custom_target(
  tests ALL
  DEPENDS ${TEST_TARGETS}
  COMMENT "Building test deps"
)

# Add a custom target to build all .hsaco files in the output directory
add_custom_target(
  all_hsaco_kernels ALL
  DEPENDS ${HSACO_TARGETS}
  COMMENT "Building and copying all .hsaco kernels to ${CUSTOM_KERNEL_OUTPUT_DIR}"
)
