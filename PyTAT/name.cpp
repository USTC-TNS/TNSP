/**
 * Copyright (C) 2020-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "PyTAT.hpp"

namespace TAT {
   void set_name(py::module_& tat_m) {
#ifdef TAT_USE_FAST_NAME
      py::class_<DefaultName>(tat_m, "Name", "Name used in edge of tensor")
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def_readonly("hash", &DefaultName::hash)
            .def("__hash__",
                 [](const DefaultName& name) {
                    return name.hash;
                 })
            .def_property_readonly(
                  "name",
                  [](const DefaultName& name) {
                     return static_cast<std::string>(name);
                  })
            .def(implicit_init<DefaultName, std::string>(), py::arg("name"), "Name with specified name")
            .def("__repr__",
                 [](const DefaultName& name) {
                    return "Name[" + static_cast<std::string>(name) + "]";
                 })
            .def("__str__", [](const DefaultName& name) {
               return static_cast<std::string>(name);
            });
#endif
   }
} // namespace TAT
