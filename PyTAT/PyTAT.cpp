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

#include <pybind11/numpy.h>

#include "PyTAT.hpp"

namespace TAT {

   template<typename Symmetry>
   struct Configuration {
      int L1, L2;
      std::vector<std::optional<std::pair<Symmetry, int>>> data;

      Configuration(int L1, int L2) : L1(L1), L2(L2) {
         data.resize(L1 * L2);
      }

      void setitem(const std::tuple<int, int, int>& key, const std::optional<std::pair<Symmetry, int>>& value) {
         const auto& [l1, l2, orbit] = key;
         auto offset = (orbit * L1 + l1) * L2 + l2;
         while (offset >= data.size()) {
            data.resize(data.size() * 2);
         }
         data[offset] = value;
      }

      std::optional<std::pair<Symmetry, int>> getitem(const std::tuple<int, int, int>& key) const {
         const auto& [l1, l2, orbit] = key;
         auto offset = (orbit * L1 + l1) * L2 + l2;
         if (offset >= data.size()) {
            return {};
         } else {
            return data[offset];
         }
      }
   };

   template<typename Symmetry>
   auto declare_configuration(py::module_& symmetry_m, const char* name) {
      using C = Configuration<Symmetry>;
      return py::class_<C>(symmetry_m, "Configuration", (std::string(name) + "SymmetryConfiguration").c_str())
            .def(py::init([](const int& L1, const int& L2) {
                    return C(L1, L2);
                 }),
                 "Create an empty configuration from an abstract state.")
            .def("__getitem__",
                 [](const C& self, const std::tuple<int, int, int>& key) {
                    return self.getitem(key);
                 })
            .def("__setitem__",
                 [](C& self, const std::tuple<int, int, int>& key, const int& value) {
                    self.setitem(key, std::make_pair(Symmetry(), value));
                 })
            .def("__setitem__",
                 [](C& self, const std::tuple<int, int, int>& key, const std::optional<std::pair<Symmetry, int>>& value) {
                    self.setitem(key, value);
                 })
            .def(
                  "copy",
                  [](const C& self) {
                     auto result = C(self.L1, self.L2);
                     result.data = self.data;
                     return result;
                  },
                  "Copy the configuration")
            .def_static(
                  "export_orbit0",
                  [](const std::vector<const C*>& configurations) {
                     auto configuration_number = configurations.size();
                     const auto& config0 = *configurations[0];
                     auto size = config0.L1 * config0.L2;
                     auto result =
                           py::array_t<int>({Py_ssize_t(configuration_number), Py_ssize_t(1), Py_ssize_t(config0.L1), Py_ssize_t(config0.L2)});
                     auto pointer = static_cast<int*>(result.request().ptr);
                     for (auto c = 0; c < configuration_number; c++) {
                        for (auto i = 0; i < size; i++) {
                           pointer[c * size + i] = configurations[c]->data[i]->second;
                        }
                     }
                     return result;
                  },
                  "Export configuration of orbit 0 as an array")
            .def_static(
                  "get_hat",
                  [](const std::vector<const C*>& configurations,
                     const std::vector<std::tuple<int, int, int>>& sites,
                     const std::vector<int>& physics_dims) {
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
                           p += configurations[c]->getitem(sites[i])->second;
                        }
                        pointer[c * total_physics_dim + p] = 1;
                     }
                     return hat;
                  });
   }

   // declare some function defined in other file
   void set_name(py::module_&);
   void set_navigator(py::module_&);
   void set_random(py::module_&);
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
   std::function<void()> dealing_Tensor_##SYM##_##SCALARSHORT( \
         py::module_& symmetry_m, \
         const std::string& scalar_short_name, \
         const std::string& scalar_name, \
         const std::string& symmetry_short_name);
   TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

   PYBIND11_MODULE(TAT, tat_m) {
      tat_m.doc() = "TAT is A Tensor library!";
      tat_m.attr("version") = version;
      tat_m.attr("information") = information;
      // random
      set_random(tat_m);
      // name
      set_name(tat_m);
      // symmetry and edge and edge segment
      auto No_m = tat_m.def_submodule("No");
      declare_symmetry<NoSymmetry>(No_m, "No").def(py::init<>());
      declare_edge<NoSymmetry, void, false>(No_m, "No");
      declare_edge<NoSymmetry, void, false, edge_segments_t>(No_m, "No");
      declare_configuration<NoSymmetry>(No_m, "No");

      auto Z2_m = tat_m.def_submodule("Z2");
      declare_symmetry<Z2Symmetry>(Z2_m, "Z2")
            .def(py::init<>())
            .def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2"))
            .def_property_readonly("z2", [](const Z2Symmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<Z2Symmetry, Z2, false>(Z2_m, "Z2");
      declare_edge<Z2Symmetry, Z2, false, edge_segments_t>(Z2_m, "Z2");
      declare_configuration<Z2Symmetry>(Z2_m, "Z2");

      auto U1_m = tat_m.def_submodule("U1");
      declare_symmetry<U1Symmetry>(U1_m, "U1")
            .def(py::init<>())
            .def(implicit_init<U1Symmetry, U1>(), py::arg("u1"))
            .def_property_readonly("u1", [](const U1Symmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<U1Symmetry, U1, false>(U1_m, "U1");
      declare_edge<U1Symmetry, U1, false, edge_segments_t>(U1_m, "U1");
      declare_configuration<U1Symmetry>(U1_m, "U1");

      auto Fermi_m = tat_m.def_submodule("Fermi");
      declare_symmetry<FermiSymmetry>(Fermi_m, "Fermi")
            .def(py::init<>())
            .def(implicit_init<FermiSymmetry, U1>(), py::arg("fermi"))
            .def_property_readonly("fermi", [](const FermiSymmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<FermiSymmetry, U1, false>(Fermi_m, "Fermi");
      declare_edge<FermiSymmetry, U1, false, edge_segments_t>(Fermi_m, "Fermi");
      declare_configuration<FermiSymmetry>(Fermi_m, "Fermi");

      auto FermiZ2_m = tat_m.def_submodule("FermiZ2");
      declare_symmetry<FermiZ2Symmetry>(FermiZ2_m, "FermiZ2")
            .def(py::init<>())
            .def(py::init<U1, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_init<FermiZ2Symmetry, const std::tuple<U1, Z2>&>([](const std::tuple<U1, Z2>& p) {
                    return std::make_from_tuple<FermiZ2Symmetry>(p);
                 }),
                 py::arg("tuple_of_fermi_z2"))
            .def_property_readonly(
                  "fermi",
                  [](const FermiZ2Symmetry& symmetry) {
                     return std::get<0>(symmetry);
                  })
            .def_property_readonly("z2", [](const FermiZ2Symmetry& symmetry) {
               return std::get<1>(symmetry);
            });
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true>(FermiZ2_m, "FermiZ2");
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true, edge_segments_t>(FermiZ2_m, "FermiZ2");
      declare_configuration<FermiZ2Symmetry>(FermiZ2_m, "FermiZ2");

      auto FermiU1_m = tat_m.def_submodule("FermiU1");
      declare_symmetry<FermiU1Symmetry>(FermiU1_m, "FermiU1")
            .def(py::init<>())
            .def(py::init<U1, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_init<FermiU1Symmetry, const std::tuple<U1, U1>&>([](const std::tuple<U1, U1>& p) {
                    return std::make_from_tuple<FermiU1Symmetry>(p);
                 }),
                 py::arg("tuple_of_fermi_u1"))
            .def_property_readonly(
                  "fermi",
                  [](const FermiU1Symmetry& symmetry) {
                     return std::get<0>(symmetry);
                  })
            .def_property_readonly("u1", [](const FermiU1Symmetry& symmetry) {
               return std::get<1>(symmetry);
            });
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true>(FermiU1_m, "FermiU1");
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true, edge_segments_t>(FermiU1_m, "FermiU1");
      declare_configuration<FermiU1Symmetry>(FermiU1_m, "FermiU1");

      auto Parity_m = tat_m.def_submodule("Parity");
      declare_symmetry<ParitySymmetry>(Parity_m, "Pariry")
            .def(py::init<>())
            .def(implicit_init<ParitySymmetry, Z2>(), py::arg("parity"))
            .def_property_readonly("parity", [](const ParitySymmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<ParitySymmetry, Z2, false>(Parity_m, "Parity");
      declare_edge<ParitySymmetry, Z2, false, edge_segments_t>(Parity_m, "Parity");
      declare_configuration<ParitySymmetry>(Parity_m, "Parity");

      // tensor
      std::vector<std::function<void()>> define_tensor;
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
   define_tensor.push_back(dealing_Tensor_##SYM##_##SCALARSHORT(SYM##_m, #SCALARSHORT, #SCALAR, #SYM));
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      for (const auto& f : define_tensor) {
         f();
      }

      // get tensor
      set_navigator(tat_m);
      // normal = no
      tat_m.attr("Normal") = tat_m.attr("No");

      at_exit.release();
   }
} // namespace TAT
