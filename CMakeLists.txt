cmake_minimum_required(VERSION 3.10)
project(DNLP_Diff_Engine C)

set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/expr.c
    src/elementwise/exp.c
    src/elementwise/log.c
    src/affine/variable.c
    src/affine/constant.c
    src/affine/add.c
)

# Create library
add_library(dnlp_diff ${SOURCES})
target_link_libraries(dnlp_diff m)

# Enable testing
enable_testing()

# Single test executable that links all test source files
add_executable(test_all
    tests/forward_pass/all_tests.c
    tests/forward_pass/test_helpers.c
    tests/forward_pass/affine/test_variable_constant.c
    tests/forward_pass/affine/test_add.c
    tests/forward_pass/elementwise/test_exp.c
    tests/forward_pass/elementwise/test_log.c
    tests/forward_pass/composite/test_composite.c
)
target_link_libraries(test_all dnlp_diff)
add_test(NAME AllTests COMMAND test_all)
