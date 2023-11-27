/**
 * Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <optional>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Configuration {
    int L1, L2;
    std::vector<std::optional<int>> data;

    Configuration(int L1, int L2) : L1(L1), L2(L2) {
        data.resize(L1 * L2);
    }

    void setitem(const std::tuple<int, int, int>& key, const std::optional<int>& value) {
        const auto& [l1, l2, orbit] = key;
        auto offset = (orbit * L1 + l1) * L2 + l2;
        while (offset >= data.size()) {
            data.resize(data.size() * 2);
        }
        data[offset] = value;
    }

    std::optional<int> getitem(const std::tuple<int, int, int>& key) const {
        const auto& [l1, l2, orbit] = key;
        auto offset = (orbit * L1 + l1) * L2 + l2;
        if (offset >= data.size()) {
            return {};
        } else {
            return data[offset];
        }
    }
};

auto dealing_configuration(py::module_& m) {
    using C = Configuration;
    return py::class_<C>(m, "Configuration", "Configuration for a state on square lattice")
        .def(py::init<int, int>(), "Create an empty configuration from an abstract state.", py::arg("L1"), py::arg("L2"))
        .def("__getitem__", &C::getitem)
        .def("__setitem__", &C::setitem)
        .def(
            "copy",
            [](const C& self) {
                auto result = C(self.L1, self.L2);
                result.data = self.data;
                return result;
            },
            "Copy the configuration"
        )
        .def_static(
            "export_orbit0",
            [](const std::vector<const C*>& configurations) {
                auto configuration_number = configurations.size();
                const auto& config0 = *configurations[0];
                auto size = config0.L1 * config0.L2;
                auto result = py::array_t<int>({Py_ssize_t(configuration_number), Py_ssize_t(1), Py_ssize_t(config0.L1), Py_ssize_t(config0.L2)});
                auto pointer = static_cast<int*>(result.request().ptr);
                for (auto c = 0; c < configuration_number; c++) {
                    for (auto i = 0; i < size; i++) {
                        pointer[c * size + i] = configurations[c]->data[i].value();
                    }
                }
                return result;
            },
            "Export configuration of orbit 0 as an array"
        )
        .def_static(
            "get_hat",
            [](const std::vector<const C*>& configurations, const std::vector<std::tuple<int, int, int>>& sites, const std::vector<int>& physics_dims
            ) {
                auto configuration_number = configurations.size();
                auto physics_edge_number = sites.size();
                auto total_physics_dim = 1;
                for (auto d : physics_dims) {
                    total_physics_dim *= d;
                }
                auto hat = py::array_t<int>({Py_ssize_t(configuration_number), Py_ssize_t(total_physics_dim)});
                auto pointer = static_cast<int*>(hat.request().ptr);
                std::fill(pointer, pointer + configuration_number * total_physics_dim, 0);
                for (auto c = 0; c < configuration_number; c++) {
                    auto p = 0;
                    for (auto i = 0; i < physics_edge_number; i++) {
                        p *= physics_dims[i];
                        p += configurations[c]->getitem(sites[i]).value();
                    }
                    pointer[c * total_physics_dim + p] = 1;
                }
                return hat;
            }
        );
}

PYBIND11_MODULE(tetraux, m) {
    m.doc() = "tetraux contains some auxiliary function and class used by tetragono.";
    dealing_configuration(m);
}
