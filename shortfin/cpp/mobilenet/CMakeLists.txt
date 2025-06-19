add_executable(mnet)
target_link_libraries(mnet
  PUBLIC
    iree_base_base
  PRIVATE
    ${SHORTFIN_LINK_LIBRARY_NAME}
)
target_sources(mnet PUBLIC driver.cc mobilenet.cc mobilenet.h)

set_target_properties(mnet PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
