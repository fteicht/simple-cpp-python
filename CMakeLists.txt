cmake_minimum_required(VERSION 3.10.2)

# Set CMAKE_BUILD_TYPE to Release by default.
# Must be done before calling project()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" Release Debug RelWithDebInfo MinSizeRel)
endif()

project(simple_cpp_python
        VERSION 0.1.0
        DESCRIPTION "A minimal project showing how to configure and code a project mixing C++ and Python"
        HOMEPAGE_URL "https://github.com/fteicht/simple-cpp-python/"
        LANGUAGES CXX)

find_package(TBB REQUIRED)  # TBB required by GCC's implementation of C++-17
find_package(pybind11 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Backward)

# C++-17 required for high-level code parallelization.
# Unfortunately pybind11 interferes with CMake; we cannot call
# target_compile_features() as is done with test-cpp-inner-loop,
# and have to set PYBIND11_CPP_STANDARD variable before calling
# pybind11_add_module.  Ideally pybind11 should have an option
# to not enforce C++14.
set(PYBIND11_CPP_STANDARD --std=c++17)
pybind11_add_module(__simple_cpp_python package.cc)
target_link_libraries(__simple_cpp_python PRIVATE ${TBB_IMPORTED_TARGETS})

include(CTest)
include(Catch)

add_executable(test-cpp-inner-loop tests.cc)
target_compile_features(test-cpp-inner-loop PRIVATE cxx_std_17)
target_link_libraries(test-cpp-inner-loop PRIVATE
                      Catch2::Catch2 pybind11::embed ${TBB_IMPORTED_TARGETS})
if (TARGET Backward::Backward)
    target_compile_definitions(test-cpp-inner-loop PRIVATE HAVE_BACKWARD)
    target_link_libraries(test-cpp-inner-loop PRIVATE Backward::Backward)
endif ()
catch_discover_tests(test-cpp-inner-loop)
