
cmake_minimum_required(VERSION 3.15)
project(DNLP_Diff_Engine C)


# Prefer OpenBLAS unless user overrides BLA_VENDOR
# if(NOT DEFINED BLA_VENDOR)
#   set(BLA_VENDOR "OpenBLAS" CACHE STRING "Preferred BLAS vendor" FORCE)
#   message(STATUS "BLA_VENDOR not set, defaulting to OpenBLAS.")
# else()
#   message(STATUS "BLA_VENDOR set to ${BLA_VENDOR}")
# endif()

set(CMAKE_C_STANDARD 99)

#Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()


# =============================================================================
# BLAS/LAPACK Configuration (DISABLED)
# =============================================================================
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)
#
# message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
# message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
#
# set(BLAS_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# add_compile_definitions(USE_BLAS)
#
# # Optional BLAS configuration flags
# option(BLAS_64 "Use 64-bit integers for BLAS" OFF)
# option(NO_BLAS_SUFFIX "BLAS functions have no suffix" OFF)
#
# if(BLAS_64)
#   add_compile_definitions(BLAS_64)
#   message(STATUS "Using 64-bit BLAS integers")
# endif()
#
# if(NO_BLAS_SUFFIX)
#   add_compile_definitions(NO_BLAS_SUFFIX)
#   message(STATUS "Using BLAS with no function suffix")
# endif()

# Warning flags (always enabled)
add_compile_options(
    -Wall                  # Enable most warnings
    -Wextra                # Extra warnings
    -Wpedantic             # Strict ISO C compliance
    -Wshadow               # Warn about variable shadowing
    -Wformat=2             # Extra format string checks
    -Wcast-qual            # Warn about cast that removes qualifiers
    -Wcast-align           # Warn about pointer cast alignment issues
    -Wunused               # Warn about unused variables/functions
    -Wdouble-promotion     # Warn about float->double promotion
    -Wnull-dereference     # Warn about null pointer dereference
)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files - automatically gather all .c files from src/
file(GLOB_RECURSE SOURCES "src/*.c")


# Create core library
add_library(dnlp_diff ${SOURCES})
# target_link_libraries(dnlp_diff m ${BLAS_LIBRARIES})
target_link_libraries(dnlp_diff m)

# Config-specific compile options
target_compile_options(dnlp_diff PRIVATE
  $<$<CONFIG:Debug>:-g -O0>
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
  $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
)

# This is needed for clock_gettime on Linux without compiler extensions
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# Enable position-independent code for shared library compatibility
set_property(TARGET dnlp_diff PROPERTY POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Python bindings (built when using scikit-build-core)
# =============================================================================
if(SKBUILD)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

    # Create Python extension module
    Python3_add_library(_core MODULE python/bindings.c)
    target_include_directories(_core PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/python
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(_core PRIVATE dnlp_diff)

    # Install to the package directory
    install(TARGETS _core LIBRARY DESTINATION dnlp_diff_engine)
endif()

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
if(NOT SKBUILD)
    include_directories(${PROJECT_SOURCE_DIR}/tests)
    enable_testing()

    add_executable(all_tests
        tests/all_tests.c
        tests/test_helpers.c
    )
    target_link_libraries(all_tests dnlp_diff)
    add_test(NAME AllTests COMMAND all_tests)
endif()
