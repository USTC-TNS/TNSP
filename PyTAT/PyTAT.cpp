/**
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM)                                                                                     \
   void dealing_Tensor_##SCALARSHORT##SYM(                                                                                                       \
         py::module_& symmetry_m, const std::string& scalar_short_name, const std::string& scalar_name, const std::string& symmetry_short_name); \
   void dealing_MPI_##SCALARSHORT##SYM(py::class_<mpi_t>& py_mpi_t);
   TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

#ifndef TAT_PYTHON_MODULE
#define TAT_PYTHON_MODULE TAT
#endif

   PYBIND11_MODULE(TAT_PYTHON_MODULE, tat_m) {
      tat_m.doc() = "TAT is A Tensor library!";
      tat_m.attr("version") = version;
      tat_m.attr("information") = information;
      // random
      auto random_m = tat_m.def_submodule("random", "random for TAT");
      random_m.def("seed", &set_random_seed, "Set Random Seed");
      random_m.def(
            "uniform_int",
            [](int min, int max) {
               return [distribution = std::uniform_int_distribution<int>(min, max)]() mutable { return distribution(random_engine); };
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform integer");
      random_m.def(
            "uniform_real",
            [](double min, double max) {
               return [distribution = std::uniform_real_distribution<double>(min, max)]() mutable { return distribution(random_engine); };
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform real");
      random_m.def(
            "normal",
            [](double mean, double stddev) {
               return [distribution = std::normal_distribution<double>(mean, stddev)]() mutable { return distribution(random_engine); };
            },
            py::arg("mean") = 0,
            py::arg("stddev") = 1,
            "Get random normal real");
      // mpi
      auto py_mpi_t = py::class_<mpi_t>(tat_m, "mpi_t", "several functions for MPI")
                            .def_readonly_static("enabled", &mpi_t::enabled)
                            .def_readonly("rank", &mpi_t::rank)
                            .def_readonly("size", &mpi_t::size)
                            .def("__str__",
                                 [](const mpi_t& mpi) {
                                    auto out = std::stringstream();
                                    out << "[rank=" << mpi.rank << ", size=" << mpi.size << "]";
                                    return out.str();
                                 })
                            .def("__repr__",
                                 [](const mpi_t& mpi) {
                                    auto out = std::stringstream();
                                    out << "MPI[rank=" << mpi.rank << ", size=" << mpi.size << "]";
                                    return out.str();
                                 })
                            .def("print",
                                 [](const mpi_t& mpi, const py::args& args, const py::kwargs& kwargs) {
                                    if (mpi.rank == 0) {
                                       py::print(*args, **kwargs);
                                    }
                                 })
#ifdef TAT_USE_MPI
                            .def_static("barrier", &mpi_t::barrier)
#endif
            ;
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) dealing_MPI_##SCALARSHORT##SYM(py_mpi_t);
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      tat_m.attr("mpi") = mpi;
      // name
#ifndef TAT_USE_SIMPLE_NAME
      py::class_<DefaultName>(tat_m, "Name", "Name used in edge of tensor, which is just a string but stored by identical integer")
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::init<fastname_dataset_t::fast_name_id_t>(), py::arg("id"), "Name with specified id directly")
            .def_readonly("id", &DefaultName::id)
            .def("__hash__", [](const DefaultName& name) { return py::hash(py::cast(name.id)); })
            .def_static("load", [](const py::bytes& bytes) { return load_fastname_dataset(std::string(bytes)); })
            .def_static("dump", []() { return py::bytes(dump_fastname_dataset()); })
            .def_property_readonly("name", [](const DefaultName& name) { return static_cast<const std::string&>(name); })
            .def(implicit_init<DefaultName, const char*>(), py::arg("name"), "Name with specified name")
            .def("__repr__", [](const DefaultName& name) { return "Name[" + static_cast<const std::string&>(name) + "]"; })
            .def("__str__", [](const DefaultName& name) { return static_cast<const std::string&>(name); });
#endif
      auto No_m = tat_m.def_submodule("No");
      declare_symmetry<NoSymmetry>(No_m, "No").def(py::init<>());
      declare_edge<NoSymmetry, void, false>(No_m, "No");
      declare_edge<NoSymmetry, void, false, edge_map_t>(No_m, "No");
      auto Z2_m = tat_m.def_submodule("Z2");
      declare_symmetry<Z2Symmetry>(Z2_m, "Z2")
            .def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2"))
            .def_property_readonly("z2", [](const Z2Symmetry& symmetry) { return std::get<0>(symmetry); });
      declare_edge<Z2Symmetry, Z2, false>(Z2_m, "Z2");
      declare_edge<Z2Symmetry, Z2, false, edge_map_t>(Z2_m, "Z2");
      auto U1_m = tat_m.def_submodule("U1");
      declare_symmetry<U1Symmetry>(U1_m, "U1")
            .def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2"))
            .def_property_readonly("z2", [](const Z2Symmetry& symmetry) { return std::get<0>(symmetry); });
      declare_edge<U1Symmetry, U1, false>(U1_m, "U1");
      declare_edge<U1Symmetry, U1, false, edge_map_t>(U1_m, "U1");
      auto Fermi_m = tat_m.def_submodule("Fermi");
      declare_symmetry<FermiSymmetry>(Fermi_m, "Fermi")
            .def(implicit_init<FermiSymmetry, U1>(), py::arg("fermi"))
            .def_property_readonly("fermi", [](const FermiSymmetry& symmetry) { return std::get<0>(symmetry); });
      declare_edge<FermiSymmetry, U1, false>(Fermi_m, "Fermi");
      declare_edge<FermiSymmetry, U1, false, edge_map_t>(Fermi_m, "Fermi");
      auto FermiZ2_m = tat_m.def_submodule("FermiZ2");
      declare_symmetry<FermiZ2Symmetry>(FermiZ2_m, "FermiZ2")
            .def(py::init<U1, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_init<FermiZ2Symmetry, const std::tuple<U1, Z2>&>(
                       [](const std::tuple<U1, Z2>& p) { return std::make_from_tuple<FermiZ2Symmetry>(p); }),
                 py::arg("tuple_of_fermi_z2"))
            .def_property_readonly("fermi", [](const FermiZ2Symmetry& symmetry) { return std::get<0>(symmetry); })
            .def_property_readonly("z2", [](const FermiZ2Symmetry& symmetry) { return std::get<1>(symmetry); });
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true>(FermiZ2_m, "FermiZ2");
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true, edge_map_t>(FermiZ2_m, "FermiZ2");
      auto FermiU1_m = tat_m.def_submodule("FermiU1");
      declare_symmetry<FermiU1Symmetry>(FermiU1_m, "FermiU1")
            .def(py::init<U1, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_init<FermiU1Symmetry, const std::tuple<U1, U1>&>(
                       [](const std::tuple<U1, U1>& p) { return std::make_from_tuple<FermiU1Symmetry>(p); }),
                 py::arg("tuple_of_fermi_u1"))
            .def_property_readonly("fermi", [](const FermiU1Symmetry& symmetry) { return std::get<0>(symmetry); })
            .def_property_readonly("u1", [](const FermiU1Symmetry& symmetry) { return std::get<1>(symmetry); });
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true>(FermiU1_m, "FermiU1");
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true, edge_map_t>(FermiU1_m, "FermiU1");
      // tensor
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) dealing_Tensor_##SCALARSHORT##SYM(SYM##_m, #SCALARSHORT, #SCALAR, #SYM);
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      // get tensor
      tat_m.def("navigator", [tat_m](const py::args& args, const py::kwargs& kwargs) -> py::object {
         if (py::len(args) == 0 && py::len(kwargs) == 0) {
            tat_m.attr("mpi").attr("print")(information);
            return py::none();
         }
         auto text = py::str(py::make_tuple(args, kwargs));
         auto contain = [&text](const char* string) { return py::cast<bool>(text.attr("__contains__")(string)); };
         if (contain("mpi") || contain("MPI")) {
            return tat_m.attr("mpi");
         }
         std::string scalar = "";
         std::string fermi = "";
         std::string symmetry = "";
         if (contain("Fermi")) {
            fermi = "Fermi";
         }
         if (contain("Bose")) {
            if (fermi != "") {
               throw std::runtime_error("Fermi Ambiguous");
            }
         }
         if (contain("U1")) {
            symmetry = "U1";
         }
         if (contain("Z2")) {
            if (symmetry == "") {
               symmetry = "Z2";
            } else {
               throw std::runtime_error("Symmetry Ambiguous");
            }
         }
         if (contain("No")) {
            if (symmetry != "") {
               throw std::runtime_error("Symmetry Ambiguous");
            }
         }
         if (symmetry == "" && fermi == "") {
            symmetry = "No";
         }
         if (contain("complex")) {
            scalar = "Z";
         }
         if (contain("complex32")) {
            scalar = "C";
         }
         if (contain("float")) {
            if (scalar == "") {
               scalar = "D";
            } else {
               throw std::runtime_error("Scalar Ambiguous");
            }
         }
         if (contain("float32")) {
            if (scalar == "" || scalar == "D") {
               scalar = "S";
            } else {
               throw std::runtime_error("Scalar Ambiguous");
            }
         }
         if (scalar == "") {
            throw std::runtime_error("Scalar Ambiguous");
         }
         return tat_m.attr((fermi + symmetry).c_str()).attr(scalar.c_str()).attr("Tensor");
      });
      auto py_type = py::module_::import("builtins").attr("type");
      py::dict callable_type_dict;
      callable_type_dict["__call__"] = tat_m.attr("navigator");
      py::list base_types;
      base_types.append(py::type::of(tat_m));
      tat_m.attr("__class__") = py_type("CallableModule", py::tuple(base_types), callable_type_dict);
      at_exit.release();
   }
} // namespace TAT
