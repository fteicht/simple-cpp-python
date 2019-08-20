#include <execution>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <iostream>

#include <functional>
#include <numeric>
#include <cmath>

// see https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Lagrange's_trigonometric_identities

namespace py = pybind11;

double cpp_inner_loop(const unsigned int& M) {
    std::vector<double> v(M);
    std::iota(v.begin(), v.end(), 0);
    std::transform(std::execution::seq, v.begin(), v.end(), v.begin(), [](const double& k){
        return 1.0 / ((4*k + 1) * (4*k + 3));
    });
    return std::reduce(std::execution::seq, v.begin(), v.end(), 0.0, std::plus<>());
}


double cpp_outer_loop(const std::function<double (const unsigned int&)>& xf,
                      const unsigned int& N, const unsigned int& M) {
    try {
        std::vector<double> v(N);
        std::iota(v.begin(), v.end(), 0);
        std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf, &M](const auto& n){
            return std::cos((n+1) * xf(M)) + std::sin((n+1) * xf(M));
        });
        return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
    } catch (const std::exception& e) {
        py::print(py::str(e.what()));
        return -1;
    }
}


double cpp_outer_loop_gil(const std::function<double (const unsigned int&)>& xf,
                          const unsigned int& N, const unsigned int& M) {
    try {
        py::gil_scoped_release release;
        std::vector<double> v(N);
        std::iota(v.begin(), v.end(), 0);
        std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf, &M](const auto& n){
            py::gil_scoped_acquire acquire;
            return std::cos((n+1) * xf(M)) + std::sin((n+1) * xf(M));
        });
        return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
    } catch (const std::exception& e) {
        py::print(py::str(e.what()));
        return -1;
    }
}


double cpp_outer_loop_process(const std::function<void (const unsigned int&, const unsigned int&)>& xg,
                              const std::function<double (const unsigned int&)>& xf,
                              const unsigned int& N, const unsigned int& M) {
    try {
        py::gil_scoped_release release;
        std::vector<double> v(N);
        std::iota(v.begin(), v.end(), 0);
        std::for_each(std::execution::par, v.begin(), v.end(), [&xg, &M](const auto& n){
            py::gil_scoped_acquire acquire;
            xg(n, M);
        });
        std::transform(std::execution::par, v.begin(), v.end(), v.begin(), [&xf](const auto& n){
            py::gil_scoped_acquire acquire;
            return std::cos((n+1) * xf(n)) + std::sin((n+1) * xf(n));
        });
        return std::reduce(std::execution::par, v.begin(), v.end(), 0.0, std::plus<>());
    } catch (const std::exception& e) {
        py::print(py::str(e.what()));
        return -1;
    }
}


double full_cpp(const unsigned int& N, const unsigned int& M) {
    return cpp_outer_loop(cpp_inner_loop, N, M);
}
