#include "cft_revival/kernels.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
    module.doc() = "Validated C++ kernels for cft_revival";
    module.def(
        "cusp_arrival_probability",
        &cft_revival::cusp_arrival_probability,
        py::arg("low_field_t"),
        py::arg("high_field_t"));
    module.def(
        "cusp_arrival_probabilities",
        &cft_revival::cusp_arrival_probabilities,
        py::arg("low_field_t"),
        py::arg("high_field_t"));
}
