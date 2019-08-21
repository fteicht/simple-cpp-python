---
marp: true
---

<!--
$theme: gaia
class: invert -->

<!-- _class: lead invert -->

# A simple project mixing parallel Python and C++ codes

---

# We will show...

- how to configure a platform independent project mixing Python and C++ with [cmake](https://cmake.org) and [pybind11](https://github.com/pybind/pybind11)
- how to call Python from C++ and vice versa
- how to code in modern C++ as in Python
- how to write efficient parallel code mixing C++ and Python by circumventing Python's GIL
- how to unit-test C++ code

---

# Requirements

- good C++ compiler with C++-17 standard support: e.g. [GCC](https://gcc.gnu.org) or [Clang](https://clang.llvm.org)
- [CMake](https://cmake.org): platform-independent configuration tool for C++
- [pybind11](https://github.com/pybind/pybind11): high-level C++ directives to call Python from C++ and vice versa; built upon cython
- [backward](https://github.com/bombela/backward-cpp): magically adds stack trace capabilities for C++ without additional code (only if project compiled in debug mode)
- [Catch2](https://github.com/catchorg/Catch2): beautiful unit tests in C++

---

# Project architecture

```bash
simple-cpp-python
    |-- CMakeLists.txt   # project configuration
    |-- code.hh          # C++ functions code
    |-- package.cc       # export C++ functions to python
    |-- tests.cc         # C++ unit tests
    |-- script.py        # python code calling C++ code
    |-- build            # created after project configuration (see next slide)
        |-- __simple_cpp_python.cpython-37m-darwin.so # C++ python extension
```

---

# Project configuration: CMakeLists.txt

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
PROJECT(simple_cpp_python)

ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 17)  # C++-17 required for high-level code parallelization
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(TBB REQUIRED)  # TBB required by GCC's implementation of C++-17
FIND_PACKAGE(pybind11 REQUIRED)
FIND_PACKAGE(Catch2 REQUIRED)
FIND_PACKAGE(Backward)  # optional package

PYBIND11_ADD_MODULE(__simple_cpp_python main.cc)
TARGET_LINK_LIBRARIES(__simple_cpp_python PRIVATE ${TBB_IMPORTED_TARGETS})
```

---

# Configuring the project

1. Go to the directory where you installed ```simple-cpp-python```
2. Create a ```build``` directory there
3. Run the following commands (or use an IDE like visual studio code):
   ```bash
   cd build
   cmake ..  # configure the project for your platform (creates a Makefile)
   make      # compile the python extension library
   ```
You should now have a file under ```build``` called
```__simple_cpp_python.cpython-37m-darwin.so``` that you can import in Python.

---

# Selecting different python interpreters

In the previous slide, cmake via the pybind11 package (line
```FIND_PACKAGE(pybind11 REQUIRED)``` in CMakeLists.txt) automatically selects the default python interpreter of my shell.

To select a different python interpreter you need to configure with the following command:
```cmake -DPYTHON_EXECUTABLE="/path/to/my/python/interpreter" ..```

---

# Computing Lagrange's trigonometric identities

See the [Wikipedia](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Lagrange's_trigonometric_identities) page. We gonna compute:

$$ \begin{aligned}
f(\theta, N) &= \sum_{n=1}^N \sin(n \theta) + \cos(n \theta) \\
             & = \frac{1}{2} \left( \cot \left(\frac{\theta}{2}\right) - 1 + \frac{\sin \left( \left(N+\frac{1}{2}\right) \theta \right) - \cos \left( \left(N+\frac{1}{2}\right) \theta \right)}{\sin \left(\frac{\theta}{2}\right)} \right)
\end{aligned} $$
with $\theta = \frac{\pi}{8} = \sum_{k=0}^{+\infty} \frac{1}{(4k+1)(4k+3)}$

---

# Why such a dummy useless nested loops computation?

<br>
<span style="font-size:95%">
Only to show:
- the benefit of parallezing code mixing Python and C++
- how to call python (inner loop) from C++ (outer loop)

Outer loop (i.e. sum from $n=1$ to $N$) used to demonstrate parallelization (compute each element of the sum in parallel)

Overhead of parallel for loops mitigated if computation of each element of the loop takes a lot of CPU cycles --> compute $\sin(n\theta)$ and $\cos(n\theta)$ with $\theta=\frac{\pi}{8}$ as a series (inner loop).
</span>

---

# Full python code

```python
import math, operator, time, sys
from pathos import multiprocessing
from functools import reduce

def python_inner_loop(M):  # computes pi/8 as a sequential series
    return reduce(operator.add, [1 / ((4*k + 1) * (4*k + 3)) for k in range(M)], 0)

def python_outer_loop(xf, N, M):  # computes the parallel series of cos(n * theta) + sin(n * theta)
    pool = multiprocessing.Pool()
    v = pool.map(lambda n: math.cos((n+1) * xf(M)) + math.sin((n+1) * xf(M)), [n for n in range(N)])
    return reduce(operator.add, v, 0)

def full_python(N, M):
    return python_outer_loop(python_inner_loop, N, M)

if __name__ == "__main__":
    start = time.time()
    r = full_python(1000, 10000)
    end = time.time()
    print('full python result: ' + str(r) + ' in ' + str(end-start) + ' seconds')
    x = math.pi / 8
    r = 0.5 * ((1.0 / math.tan(x / 2)) - 1 + (math.sin(x * (N + 0.5)) - math.cos(x * (N + 0.5))) / math.sin(x / 2))
    print('exact result: ' + str(r))
```

---

# Full python code (cont.)

Result:
```bash
$ python script.py
full python result: 4.046217839811811 in 1.1689767837524414 seconds
exact result: 4.027339492125908
```

---

# Full C++ version

```cpp
double cpp_inner_loop(const unsigned int& M) {
    std::vector<double> v(M);
    std::iota(v.begin(), v.end(), 0);  // fills the vector v with [0, 1, 2, ..., M-1]
    std::transform(std::execution::seq, v.begin(), v.end(), v.begin(), [](const double& k){ // sequential loop
        return 1.0 / ((4*k + 1) * (4*k + 3));
    });
    return std::reduce(std::execution::seq, v.begin(), v.end(), 0.0, std::plus<>());
}

double cpp_outer_loop(const std::function<double (const unsigned int&)>& xf, const unsigned int& N, const unsigned int& M) {
    std::vector<double> v(N);
    std::iota(v.begin(), v.end(), 0);  // fills the vector v with [0, 1, 2, ..., N-1]
    std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf, &M](const auto& n){ // parallel loop
        return std::cos((n+1) * xf(M)) + std::sin((n+1) * xf(M));
    });
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
}

double full_cpp(const unsigned int& N, const unsigned int& M) {
    return cpp_outer_loop(cpp_inner_loop, N, M);
}
```
<span style="font-size:75%">
Note how std::transform() is called with std::execution::seq or std::execution::par.
</span>

---

# Exporting the C++ code to python

In ```package.cc```:
```cpp
#include "code.hh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(__simple_cpp_python, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("full_cpp", &full_cpp, "A function which computes the sum of
                                  Lagrangian's trigonometric identities - full c++");
}
```
<span style="font-size:75%">
It will compile a python extension library called __simple_cpp_python.cpython-37m-darwin.so (on MacOS) containing the full_cpp() python function.
</span>

---

# Calling the C++ code from python

In ```script.py```:
```python
import build.__simple_cpp_python as wow

if __name__ == "__main__":
    start = time.time()
    r = wow.full_cpp(1000, 10000)
    end = time.time()
    print('full c++ result: ' + str(r) + ' in ' + str(end-start) + ' seconds')
```

---

# Running the script and comparing full Python vs full C++

Result:
```bash
$ python script.py
full python result: 4.046217839811811 in 1.1679670810699463 seconds
full c++ result: 4.046217839811809 in 0.008682012557983398 seconds
exact result: 4.027339492125908
```
On this example **pure C++ is 134.53x faster than pure Python**; both using a sequential inner loop and a parallel outer loop.

---

# Mixing C++ (outer loop) and Python (inner loop)

In ```code.hh``` (```xf``` will be a Python lambda function when calling ```cpp_outer_loop_gil``` from Python):
```
double cpp_outer_loop_gil(const std::function<double (const unsigned int&)>& xf,
                          const unsigned int& N, const unsigned int& M) {
    py::gil_scoped_release release; // Release Python's GIL to enable OS multithreading
    std::vector<double> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf, &M](const auto& n){
        py::gil_scoped_acquire acquire;  // Acquire the GIL to prevent Python race conditions
        return std::cos((n+1) * xf(M)) + std::sin((n+1) * xf(M));  // xf() calls a Python lambda
    });
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
}
```

---

# Calling the C++ function from Python

In ```script.py```:
```python
import build.__simple_cpp_python as wow

if __name__ == "__main__":
    start = time.time()
    r = wow.cpp_outer_loop_gil(lambda M: python_inner_loop(M), 1000, 10000)
    end = time.time()
    print('mixed result: ' + str(r) + ' in ' + str(end-start) + ' seconds')
```
In short: Python calls a C++ function that calls a long-running Python function in a parallel loop.

---

# Running the script

Result:
```bash
$ python script.py
full python result: 4.046217839811811 in 1.239090919494629 seconds
full c++ result: 4.046217839811811 in 0.009485006332397461 seconds
mixed result: 4.046217839811811 in 4.09197211265564 seconds
exact result: 4.027339492125908
```
The mixed C++/Python code is **3.3x slower** than the pure Python code! (and 431x slower than the pure C++ code)
That's because of **Python's GIL** that acts like a mutex that forces to sequentialize the C++ parallel loop...

---

# Analysis

- Python's GIL forces the outer C++ parallel loop to call each inner element in sequence
- Not happening in pure Python because full python code parallelism is implemented using *multiprocessing* instead of *multithreading* (thus circumventing the GIL)
- Solution: **call each inner Python function from a different Python process!** (but each Python inner loop running in sequence as previously)

Idea: **1 C++ thread <=> 1 Python process**

---

# C++ code calling concurrent python processes in parallel

```cpp
double cpp_outer_loop_process(const std::function<void (const unsigned int&, const unsigned int&)>& xg,
                              const std::function<double (const unsigned int&)>& xf,
                              const unsigned int& N, const unsigned int& M) {
    py::gil_scoped_release release;
    std::vector<double> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::for_each(std::execution::par, v.begin(), v.end(), [&xg, &M](const auto& n){
        py::gil_scoped_acquire acquire;
        xg(n, M);  // Call to Python function that launches a detached process computing the inner loop
    });
    std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf](const auto& n){
        py::gil_scoped_acquire acquire;
        return std::cos((n+1) * xf(n)) + std::sin((n+1) * xf(n));  // xf(n) gets the result of xg(n, M)
    });
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
}
```

---

# Calling the C++ function from Python

```python
def mixed_op(pool, lr, n, M):  # this our xg() viewed from C++
    lr[n] = pool.apply_async(lambda : python_inner_loop(M))  # GIL freed after launch

def mixed_get(lr, n):  # this our xf() viewed from C++
    return lr[n].get()  # blocks the process until the result is ready

if __name__ == "__main__":
    start = time.time()
    pool = multiprocessing.Pool()
    lr = [None]*N  # lr[] saves the result of each detached process
    r = wow.cpp_outer_loop_process(lambda n, M : mixed_op(pool, lr, n, M),
                                   lambda n : mixed_get(lr, n), 1000, 10000)
    end = time.time()
    print('mixed result (process): ' + str(r) + ' in ' + str(end-start) + ' seconds')
```
<span style="font-size:70%">
The GIL blocks the (only) main Python thread just to launch processes and to get their results.
</span>

---

# Running the script

```bash
$ python script.py
full python result: 4.046217839811811 in 1.1558029651641846 seconds
full c++ result: 4.046217839811814 in 0.00848531723022461 seconds
mixed result (GIL): 4.046217839811811 in 4.034562110900879 seconds
mixed result (process): 4.046217839811811 in 0.6684191226959229 seconds
exact result: 4.027339492125908
```
The mixed C++/Python process-based implementation is now **1.73x faster than the full parallel python** one (but still 78.77x slower than the full parallel C++ one...). Improvements expected to increase as inner Python code takes more CPU time.

---

# Performance analysis

Speedup vs full (parallel) Python: the higher the better (log scale)

| ![width:550px](comparison_M.png) | ![width:550px](comparison_N.png) |
|---------------------------|---------------------------|

---

# Using native python objects

```cpp
double cpp_outer_loop_process(const std::function<void (const unsigned int&, const py::dict&)>& xg,
                              const std::function<double (const unsigned int&)>& xf,
                              const unsigned int& N, const py::dict& d) {
    py::gil_scoped_release release;
    std::vector<double> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::for_each(std::execution::par, v.begin(), v.end(), [&xg, &M](const auto& n){
        py::gil_scoped_acquire acquire;
        xg(n, d);  // Work with a native python dictionary d (no copy)
    });
    std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf](const auto& n){
        py::gil_scoped_acquire acquire;
        return std::cos((n+1) * xf(n)) + std::sin((n+1) * xf(n));  // xf(n) gets the result of xg(n, M)
    });
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
}
```

---

# C++ templates: one generic outer loop for either Python or C++ inner loop

```cpp
template <typename ExecutionModel, typename InnerType>  // declare generic types
double cpp_outer_loop_process(const std::function<void (const unsigned int&, const InnerType&)>& xg,
                              const std::function<double (const unsigned int&)>& xf,
                              const unsigned int& N, const InnerType& d, const ExecutionModel& em) {
    typename ExecutionModel::Release release;  // py::gil_scoped_release if xg() is a Python lambda otherwise empty class
    std::vector<double> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::for_each(std::execution::par, v.begin(), v.end(), [&xg, &M](const auto& n){
        typename ExecutionModel::Acquire acquire;  // py::gil_scoped_acquire if xg() is a Python lambda otherwise empty class
        xg(n, d);  // d is a generic type (C++ type or e.g. python native type via pybind11)
    });
    std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf](const auto& n){
        typename ExecutionModel::Acquire acquire;  // py::gil_scoped_acquire if xg() is a Python lambda otherwise empty class
        return std::cos((n+1) * xf(n)) + std::sin((n+1) * xf(n));  // xf(n) gets the result of xg(n, M)
    });
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
}
```
<span style="font-size:60%">
ExecutionModel is a generic type that must define Release and Acquire classes (empty in case of pure C++).
</span>

---

# Using the generic C++ outer loop with C++ inner function (from C++)

```cpp
struct CPPExecutionModel {
    struct Acquire {};
    struct Release {};
};

void cpp_xg(const unsigned int& n, const std::map<std::string, double>& m) {...};
double cpp_xf(const unsigned int& n) {...};
unsigned int N = ...;
std::map<std::string, double> m = ...;

double r = cpp_outer_loop_process(cpp_xg, cpp_xf, N, m, CPPExecutionModel());
```

---

# Using the generic C++ outer loop with Python inner function (from Python)

```cpp
struct PYExecutionModel {
    typedef py::gil_scoped_acquire Acquire;
    typedef py::gil_scoped_release Release;
};

double export_cpp_outer_loop_process(const std::function<void (const unsigned int&, const py::dict&)>& xg,
                                     const std::function<double (const unsigned int&)>& xf,
                                     const unsigned int& N, const py::dict& d) {
    return cpp_outer_loop_process(xg, xf, N, d, PYExecutionModel());  // instantiate generic C++ outer loop function for Python
}

PYBIND11_MODULE(__simple_cpp_python, m) {
    m.def("cpp_outer_loop_process", &export_cpp_outer_loop_process);  // Export the specialized function to Python
}
```
```python
r = export_cpp_outer_loop_process(lambda n, d: ..., lambda n: ..., N, my_dict)  # use with native python dict
```

---

# Unit-testing the C++ code

Append the following lines to your ```CMakeLists.txt```:
```cmake
INCLUDE(CTest)
INCLUDE(Catch)

ADD_EXECUTABLE(test-cpp-inner-loop tests.cc)
TARGET_INCLUDE_DIRECTORIES(test-cpp-inner-loop PRIVATE ${PYTHON_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(test-cpp-inner-loop PRIVATE
                          Catch2::Catch2 ${TBB_IMPORTED_TARGETS} ${PYTHON_LIBRARIES})
CATCH_DISCOVER_TESTS(test-cpp-inner-loop)
```
It will add a unit test program called test-cpp-inner-loop.
Catch2 takes care of automatically finding your unit tests in ```tests.cc```

---

# Unit-testing the C++ code (cont.)

Contents of ```tests.cc```:
```cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include "code.hh"

TEST_CASE("Test the approximate computation of PI/8.0", "[PI computation]") {  // declare a new unit test
    double r1 = cpp_inner_loop(10);
    double r2 = cpp_inner_loop(100);
    double r3 = cpp_inner_loop(1000);
    double v = std::acos(-1.0) / 8.0;
    REQUIRE( std::fabs(v - r1) < 1e-1 );
    REQUIRE( std::fabs(v - r2) < 1e-3 );
    REQUIRE( std::fabs(v - r3) < 1e-5 );
}
 ```
<span style="font-size:65%">
You can write scenario tests, check values or exceptions, write conditonal tests, and much more...
</span>

---

# Unit test result

```bash
$ build/test-cpp-inner-loop
test-cpp-inner-loop is a Catch v2.9.2 host application.
Run with -? for options
-------------------------------------------------------------------------------
Test the approximate computation of PI/8.0
-------------------------------------------------------------------------------
/Users/teichteil_fl/Projects/simple-cpp-python/tests.cc:6
...............................................................................
/Users/teichteil_fl/Projects/simple-cpp-python/tests.cc:13: FAILED:
  REQUIRE( std::fabs(v - r3) < 1e-5 )
with expansion:
  0.0000625 < 0.00001
===============================================================================
test cases: 1 | 1 failed
assertions: 3 | 2 passed | 1 failed
```

---

# Finally: replace segfault message by useful stack trace message

Link your target against Backward libraries in ```CMakeLists.txt```, e.g.:
```cmake
TARGET_LINK_LIBRARIES(test-cpp-inner-loop PRIVATE Backward::Backward
                          Catch2::Catch2 ${TBB_IMPORTED_TARGETS} ${PYTHON_LIBRARIES})
```
Add the following line at the top of your source file, e.g. in ```tests.cc```:
```cpp
include "backward.hpp"
namespace backward { backward::SignalHandling sh; }
```

---

# Replace segfault message by useful stack trace message

Add a silly test that should produce a segfault (:warning: bad old school C++ coding style from the 90s on top of a silly bug!), e.g.:
```cpp
TEST_CASE("Stupid test to demonstrate Backward", "[backward]") {
    int* v = nullptr;  // v is a null pointer
    REQUIRE( v[0] == 0 );  // crash: Backward provides a stack trace 
}
```
*Note: we put this intentional crash test in a unit test for simplicity but Backward can trace any C++ code of your project*

---

# Backward stack trace result

Instead of an ungentle *sefgault* message we get a more informative trace (well, depend on the context) that helps debug your program:
```bash
Stack trace (most recent call last):
#2    Object "libsystem_platform.dylib", at 0x7fff5932fb3c, in _sigtramp + 28
#1    Object "test-cpp-inner-loop", at 0x10ae52478, in backward::SignalHandling::sig_handler(int, __siginfo*, void*) + 8
#0    Object "test-cpp-inner-loop", at 0x10ae5207d, in backward::SignalHandling::handleSignal(int, __siginfo*, void*) + 77
```
Without Backward we would simply get a message like:
```bash
[1]    46461 segmentation fault  build/test-cpp-inner-loop
```
Note that Backward is nearly non-intrusive in your code.