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
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
   void dealing_##SCALARSHORT##SYM(                          \
         py::module_& tensor_m,                              \
         py::module_& block_m,                               \
         const std::string& scalar_short_name,               \
         const std::string& scalar_name,                     \
         const std::string& symmetry_short_name);
   TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

#ifdef TAT_PYTHON_MODULE
   PYBIND11_MODULE(TAT_PYTHON_MODULE, tat_m) {
#else
   PYBIND11_MODULE(TAT, tat_m) {
#endif
      tat_m.doc() = "TAT is A Tensor library!";
      tat_m.attr("version") = version;
      tat_m.attr("information") = information;
      auto internal_m = tat_m.def_submodule("_internal", "internal information of TAT");
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
      py::class_<mpi_t>(internal_m, "mpi_t", "several functions for MPI")
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
#ifdef TAT_USE_MPI
            .def_static("barrier", &mpi_t::barrier)
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM)                                                                                       \
   .def_static("send", &mpi_t::send<Tensor<SCALAR, SYM##Symmetry>>)                                                                                \
         .def_static("receive", &mpi_t::receive<Tensor<SCALAR, SYM##Symmetry>>)                                                                    \
         .def("send_receive", &mpi_t::send_receive<Tensor<SCALAR, SYM##Symmetry>>)                                                                 \
         .def("broadcast", &mpi_t::broadcast<Tensor<SCALAR, SYM##Symmetry>>)                                                                       \
         .def("reduce",                                                                                                                            \
              [](const mpi_t& self,                                                                                                                \
                 const Tensor<SCALAR, SYM##Symmetry>& value,                                                                                       \
                 const int root,                                                                                                                   \
                 std::function<Tensor<SCALAR, SYM##Symmetry>(const Tensor<SCALAR, SYM##Symmetry>&, const Tensor<SCALAR, SYM##Symmetry>&)> func) {  \
                 return self.reduce(value, root, func);                                                                                            \
              })                                                                                                                                   \
         .def("summary", [](const mpi_t& self, const Tensor<SCALAR, SYM##Symmetry>& value, const int root) {                                       \
            return self.reduce(value, root, [](const Tensor<SCALAR, SYM##Symmetry>& a, const Tensor<SCALAR, SYM##Symmetry>& b) { return a + b; }); \
         })
                  TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
#endif
            .def("print", [](const mpi_t& mpi, const py::args& args, const py::kwargs& kwargs) {
               if (mpi.rank == 0) {
                  py::print(*args, **kwargs);
               }
            });
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
            .def(py::init<fast_name_dataset_t::FastNameId>(), py::arg("id"), "Name with specified id directly")
            .def_readonly("id", &DefaultName::id)
            .def("__hash__", [](const DefaultName& name) { return py::hash(py::cast(name.id)); })
            .def_static("load", [](const py::bytes& bytes) { return load_fast_name_dataset(std::string(bytes)); })
            .def_static("dump", []() { return py::bytes(dump_fast_name_dataset()); })
            .def_property_readonly("name", [](const DefaultName& name) { return static_cast<const std::string&>(name); })
            .def(implicit_init<DefaultName, const char*>(), py::arg("name"), "Name with specified name")
            .def("__repr__", [](const DefaultName& name) { return "Name[" + static_cast<const std::string&>(name) + "]"; })
            .def("__str__", [](const DefaultName& name) { return static_cast<const std::string&>(name); });
#endif
      // symmetry
      auto symmetry_m = internal_m.def_submodule("Symmetry", "All kinds of symmetries for TAT");
      declare_symmetry<NoSymmetry>(symmetry_m, "No").def(py::init<>());
      declare_symmetry<Z2Symmetry>(symmetry_m, "Z2").def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2")).def_readonly("z2", &Z2Symmetry::z2);
      declare_symmetry<U1Symmetry>(symmetry_m, "U1").def(implicit_init<U1Symmetry, U1>(), py::arg("u1")).def_readonly("u1", &U1Symmetry::u1);
      declare_symmetry<FermiSymmetry>(symmetry_m, "Fermi")
            .def(implicit_init<FermiSymmetry, Fermi>(), py::arg("fermi"))
            .def_readonly("fermi", &FermiSymmetry::fermi);
      declare_symmetry<FermiZ2Symmetry>(symmetry_m, "FermiZ2")
            .def(py::init<Fermi, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_init<FermiZ2Symmetry, const std::tuple<Fermi, Z2>&>(
                       [](const std::tuple<Fermi, Z2>& p) { return std::make_from_tuple<FermiZ2Symmetry>(p); }),
                 py::arg("tuple_of_fermi_z2"))
            .def_readonly("fermi", &FermiZ2Symmetry::fermi)
            .def_readonly("z2", &FermiZ2Symmetry::z2);
      declare_symmetry<FermiU1Symmetry>(symmetry_m, "FermiU1")
            .def(py::init<Fermi, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_init<FermiU1Symmetry, const std::tuple<Fermi, U1>&>(
                       [](const std::tuple<Fermi, U1>& p) { return std::make_from_tuple<FermiU1Symmetry>(p); }),
                 py::arg("tuple_of_fermi_u1"))
            .def_readonly("fermi", &FermiU1Symmetry::fermi)
            .def_readonly("u1", &FermiU1Symmetry::u1);
      // edge
      auto edge_m = internal_m.def_submodule("Edge", "Edges of all kinds of symmetries for TAT");
      declare_edge<NoSymmetry, void, false>(edge_m, "No");
      declare_edge<Z2Symmetry, Z2, false>(edge_m, "Z2");
      declare_edge<U1Symmetry, U1, false>(edge_m, "U1");
      declare_edge<FermiSymmetry, Fermi, false>(edge_m, "Fermi");
      declare_edge<FermiZ2Symmetry, std::tuple<Fermi, Z2>, true>(edge_m, "FermiZ2");
      declare_edge<FermiU1Symmetry, std::tuple<Fermi, U1>, true>(edge_m, "FermiU1");
      auto bose_edge_m = edge_m.def_submodule("NoArrow", "Edges without Arrow Even for Fermi Symmetry");
      declare_edge<NoSymmetry, void, false, BoseEdge>(bose_edge_m, "No");
      declare_edge<Z2Symmetry, Z2, false, BoseEdge>(bose_edge_m, "Z2");
      declare_edge<U1Symmetry, U1, false, BoseEdge>(bose_edge_m, "U1");
      declare_edge<FermiSymmetry, Fermi, false, BoseEdge>(bose_edge_m, "Fermi");
      declare_edge<FermiZ2Symmetry, std::tuple<Fermi, Z2>, true, BoseEdge>(bose_edge_m, "FermiZ2");
      declare_edge<FermiU1Symmetry, std::tuple<Fermi, U1>, true, BoseEdge>(bose_edge_m, "FermiU1");
      // tensor
      auto tensor_m = tat_m.def_submodule("Tensor", "Tensors for TAT");
      auto block_m = internal_m.def_submodule("Block", "Block of Tensor for TAT");
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) dealing_##SCALARSHORT##SYM(tensor_m, block_m, #SCALARSHORT, #SCALAR, #SYM);
      // declare_tensor<SCALAR, SYM##Symmetry>(tensor_m, block_m, #SCALARSHORT, #SCALAR, #SYM);
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      // get tensor
      internal_m.def("hub", [tensor_m, tat_m](const py::args& args, const py::kwargs& kwargs) -> py::object {
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
         return tensor_m.attr((scalar + fermi + symmetry).c_str());
      });
      auto py_module_type = py::type::of(tat_m);
      auto py_type = py::module_::import("builtins").attr("type");
      py::dict callable_type_dict;
      callable_type_dict["__call__"] = internal_m.attr("hub");
      py::list base_types;
      base_types.append(py_module_type);
      internal_m.attr("CallableModule") = py_type("CallableModule", py::tuple(base_types), callable_type_dict);
      tat_m.attr("__class__") = internal_m.attr("CallableModule");
      at_exit.release();
   }
} // namespace TAT
