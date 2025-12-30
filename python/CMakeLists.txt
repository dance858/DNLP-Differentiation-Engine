cmake_minimum_required(VERSION 3.20)
project(DNLP_diff_engine LANGUAGES C VERSION 0.1.0
  DESCRIPTION "Python bindings for DNLP diff engine")

if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python3")
endif()

# Find Python3 and NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION}")
message(STATUS "NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

# Verify NumPy version
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
  OUTPUT_VARIABLE NUMPY_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "NumPy version: ${NUMPY_VERSION}")

# Build core project from source
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_BINARY_DIR}/core_build)

# create Python module
add_library(bindings MODULE  ${CMAKE_CURRENT_LIST_DIR}/bindings.c)
target_include_directories(bindings
  PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../src
    ${CMAKE_CURRENT_LIST_DIR}/../src/utils
)

# Link against core and Python libraries
target_link_libraries(bindings
  PRIVATE
    dnlp_diff
    Python3::Python
    Python3::NumPy
)

# Place the module in the python source folder alongside example.py
set(PY_MODULE_OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR})
file(MAKE_DIRECTORY ${PY_MODULE_OUTPUT_DIR})

set_target_properties(bindings PROPERTIES
  PREFIX ""
  OUTPUT_NAME "DNLP_diff_engine"
  SUFFIX ".${Python3_SOABI}.so"

    # All configs → same folder
    LIBRARY_OUTPUT_DIRECTORY ${PY_MODULE_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${PY_MODULE_OUTPUT_DIR}  # Critical for Windows .pyd
)
