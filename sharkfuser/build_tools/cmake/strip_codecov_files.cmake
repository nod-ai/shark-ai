if(NOT DEFINED SHARKFUSER_IREE_SOURCE_DIR)
  message(FATAL_ERROR "SHARKFUSER_IREE_SOURCE_DIR must be defined")
endif()

set(THIRD_PARTY_DIR "${SHARKFUSER_IREE_SOURCE_DIR}/third_party")

# Find all .gcda files
file(GLOB_RECURSE GCDA_FILES "${THIRD_PARTY_DIR}/*.gcda")

# Find all .gcno files
file(GLOB_RECURSE GCNO_FILES "${THIRD_PARTY_DIR}/*.gcno")

# Combine the lists
set(CODECOV_FILES ${GCDA_FILES} ${GCNO_FILES})

message(STATUS "CODECOV_FILES: ${CODECOV_FILES}")

# Remove files
if(CODECOV_FILES)
  file(REMOVE ${CODECOV_FILES})
else()
  message(STATUS "No .gcda/.gcno files found.")
endif()
