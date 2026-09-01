cmake_minimum_required(VERSION 3.20)
project(cft_revival_native VERSION 0.1.0 LANGUAGES CXX)

option(CFT_BUILD_PYTHON "Build the pybind11 extension" ON)

add_library(cft_kernels STATIC
    native/src/kernels.cpp
)
target_include_directories(cft_kernels PUBLIC native/include)
target_compile_features(cft_kernels PUBLIC cxx_std_17)
set_target_properties(cft_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(cft_kernels PRIVATE /W4 /permissive-)
else()
    target_compile_options(cft_kernels PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CFT_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_native native/src/bindings.cpp)
    target_link_libraries(_native PRIVATE cft_kernels)
    install(TARGETS _native DESTINATION cft_revival)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cft_kernels_test native/tests/test_kernels.cpp)
    target_link_libraries(cft_kernels_test PRIVATE cft_kernels)
    add_test(NAME cft_kernels_test COMMAND cft_kernels_test)
endif()
