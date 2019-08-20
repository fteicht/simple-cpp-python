#include "code.hh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(__simple_cpp_python, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("full_cpp", &full_cpp, "A function which computes the sum of Lagrangian's trigonometric identities - full c++");
    m.def("cpp_outer_loop_gil", &cpp_outer_loop_gil, "A function which computes the sum of Lagrangian's trigonometric identities - mixed c++ / python");
    m.def("cpp_outer_loop_process", &cpp_outer_loop_process, "A function which computes the sum of Lagrangian's trigonometric identities - mixed c++ / python with multiprocessed inner python loop");
}