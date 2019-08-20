CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
PROJECT(simple_cpp_python)

ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 17)  # C++-17 required for high-level code parallelization
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(TBB REQUIRED)  # TBB required by GCC's implementation of C++-17
FIND_PACKAGE(pybind11 REQUIRED)
FIND_PACKAGE(Catch2 REQUIRED)

PYBIND11_ADD_MODULE(__simple_cpp_python package.cc)
TARGET_LINK_LIBRARIES(__simple_cpp_python PRIVATE ${TBB_IMPORTED_TARGETS})
# ADD_BACKWARD(__simple_cpp_python)
