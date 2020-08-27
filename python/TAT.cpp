/**
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//#define TAT_USE_MPI
#include "TAT/TAT.hpp"

namespace TAT {
   namespace py = ::pybind11;

   struct AtExit {
      std::vector<std::function<void()>> function_list;
      void operator()(std::function<void()> function) {
         function_list.push_back(function);
      }
      void release() {
         for (auto& function : function_list) {
            function();
         }
         function_list.resize(0);
      }
   };
   AtExit at_exit;

   template<class Type, class Args>
   auto implicit_init() {
      at_exit([]() { py::implicitly_convertible<Args, Type>(); });
      return py::init<Args>();
   }

   template<class Type, class Args, class Func>
   auto implicit_init(Func&& func) {
      at_exit([]() { py::implicitly_convertible<Args, Type>(); });
      return py::init(func);
   }

   template<class ScalarType, class Symmetry>
   void declare_mpi(py::module& m) {
      using T = Tensor<ScalarType, Symmetry>;
      m.def("send", &mpi::send<ScalarType, Symmetry>);
      m.def("receive", &mpi::receive<ScalarType, Symmetry>);
      m.def("send_receive", &mpi::send_receive<ScalarType, Symmetry>);
      m.def("broadcast", &mpi::broadcast<ScalarType, Symmetry>);
      m.def("reduce", [](const T& tensor, const int root, std::function<T(T, T)> op) { return mpi::reduce(tensor, root, op); });
      m.def("summary", &mpi::summary<ScalarType, Symmetry>);
   }

   template<class ScalarType, class Symmetry>
   auto singular_to_string(const typename Tensor<ScalarType, Symmetry>::Singular& s) {
      const auto& value = s.value; // std::map<Symmetry, vector<real_base_t<ScalarType>>>
      std::stringstream out;
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << value.begin()->second;
      } else {
         out << '{';
         bool first = true;
         for (const auto& [key, value] : value) {
            if (!first) {
               out << ',';
            } else {
               first = false;
            }
            out << key << ':' << value;
         }
         out << '}';
      }
      return out.str();
   }

   template<class ScalarType, class Symmetry>
   struct block_of_tensor {
      Tensor<ScalarType, Symmetry>* tensor;
      std::map<Name, Symmetry> position;
   };

   template<class ScalarType, class Symmetry>
   void
   declare_tensor(py::module& m, py::module& s, py::module& b, const char* name, const std::string& scalar_name, const std::string& symmetry_name) {
      using T = Tensor<ScalarType, Symmetry>;
      using S = typename T::Singular;
      using E = Edge<Symmetry>;
      using B = block_of_tensor<ScalarType, Symmetry>;
      py::class_<B>(
            b, name, ("Block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_name).c_str(), py::buffer_protocol())
            .def_buffer([](B& b) {
               auto& block = b.tensor->block(b.position);
               const Rank rank = b.tensor->names.size();
               auto dimensions = std::vector<int>(rank);
               auto leadings = std::vector<int>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = b.tensor->core->edges[i].map.at(b.position[b.tensor->names[i]]);
               }
               for (auto i = rank; i-- > 0;) {
                  if (i == rank - 1) {
                     leadings[i] = sizeof(ScalarType);
                  } else {
                     leadings[i] = leadings[i + 1] * dimensions[i + 1];
                  }
               }
               return py::buffer_info{
                     block.data(), sizeof(ScalarType), py::format_descriptor<ScalarType>::format(), rank, std::move(dimensions), std::move(leadings)};
            });
      py::class_<S>(s, name, ("Singulars in tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_name).c_str())
            .def_readonly("value", &S::value, "singular value dictionary")
            .def("__str__", [](const S& s) { return singular_to_string<ScalarType, Symmetry>(s); })
            .def("__repr__", [](const S& s) { return "Singular" + singular_to_string<ScalarType, Symmetry>(s); });
      py::class_<T>(m, name, ("Tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_name).c_str())
            .def(py::self + py::self)
            .def(ScalarType() + py::self)
            .def(py::self + ScalarType())
            .def(py::self += py::self)
            .def(py::self += ScalarType())
            .def(py::self - py::self)
            .def(ScalarType() - py::self)
            .def(py::self - ScalarType())
            .def(py::self -= py::self)
            .def(py::self -= ScalarType())
            .def(py::self * py::self)
            .def(ScalarType() * py::self)
            .def(py::self * ScalarType())
            .def(py::self *= py::self)
            .def(py::self *= ScalarType())
            .def(py::self / py::self)
            .def(ScalarType() / py::self)
            .def(py::self / ScalarType())
            .def(py::self /= py::self)
            .def(py::self /= ScalarType())
            .def("__str__",
                 [](const T& tensor) {
                    auto out = std::stringstream();
                    out << tensor;
                    return out.str();
                 })
            .def("__repr__",
                 [](const T& tensor) {
                    auto out = std::stringstream();
                    out << "Tensor";
                    out << tensor;
                    return out.str();
                 })
            .def(
                  "shape",
                  [](const T& tensor) {
                     auto out = std::stringstream();
                     out << '{' << console_green << "names" << console_origin << ':';
                     out << tensor.names;
                     out << ',' << console_green << "edges" << console_origin << ':';
                     out << tensor.core->edges;
                     out << '}';
                     return out.str();
                  },
                  "The shape of this tensor")
            .def(py::init<const std::vector<Name>&, const std::vector<E>&, bool>(),
                 py::arg("names"),
                 py::arg("edges"),
                 py::arg("auto_reverse") = false,
                 "Construct tensor with edge names and edge shapes")
            .def(implicit_init<T, std::tuple<std::vector<Name>, std::vector<E>>>(
                       [](const std::tuple<std::vector<Name>, std::vector<E>>& p) { return std::make_from_tuple<T>(p); }),
                 py::arg("names_and_edges"),
                 "Construct tensor with edge names and edge shape")
            .def(implicit_init<T, ScalarType>(), py::arg("number"), "Create rank 0 tensor with only one element")
            .def(
                  "value", [](const T& tensor) -> ScalarType { return tensor; }, "Get the only one element of a rank 0 tensor")
            .def("copy", &T::copy, "Deep copy a tensor")
            .def("same_shape", &T::same_shape, "Create a tensor with same shape")
            .def(
                  "map",
                  [](const T& tensor, std::function<ScalarType(ScalarType)>& function) { return tensor.map(function); },
                  py::arg("function"),
                  "Out-place map every element of a tensor")
            .def(
                  "transform",
                  [](T& tensor, std::function<ScalarType(ScalarType)>& function) -> T& { return tensor.transform(function); },
                  py::arg("function"),
                  "In-place map every element of a tensor",
                  py::return_value_policy::reference_internal)
            .def(
                  "set",
                  [](T& tensor, std::function<ScalarType()>& function) -> T& { return tensor.set(function); },
                  py::arg("function"),
                  "Set every element of a tensor by a function",
                  py::return_value_policy::reference_internal)
            .def(
                  "zero", [](T& tensor) -> T& { return tensor.zero(); }, "Set all element zero", py::return_value_policy::reference_internal)
            .def(
                  "test",
                  [](T& tensor, ScalarType first, ScalarType step) -> T& { return tensor.test(first, step); },
                  py::arg("first") = 0,
                  py::arg("step") = 1,
                  "Useful function generate simple data in tensor element for test",
                  py::return_value_policy::reference_internal)
            .def(
                  "block",
                  [](T& tensor, const std::map<Name, Symmetry>& position) {
                     auto numpy = py::module::import("numpy");
                     auto block = block_of_tensor<ScalarType, Symmetry>{&tensor, position};
                     return numpy.attr("array")(block, py::arg("copy") = false);
                  },
                  py::arg("dictionary_from_name_to_symmetry") = py::dict(),
                  "Get specified block data as a one dimension list",
                  py::keep_alive<0, 1>())
            .def(
                  "__getitem__",
                  [](T& tensor, const std::map<Name, typename T::EdgeInfoForGetItem>& position) { return tensor.at(position); },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"))
            .def(
                  "__setitem__",
                  [](T& tensor, const std::map<Name, typename T::EdgeInfoForGetItem>& position, const ScalarType& value) {
                     tensor.at(position) = value;
                  },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"),
                  py::arg("value"))
            .def("to_single", &T::template to<float>, "Convert to single float tensor")
            .def("to_double", &T::template to<double>, "Convert to double float tensor")
            .def("to_single_complex", &T::template to<std::complex<float>>, "Convert to single complex tensor")
            .def("to_double_complex", &T::template to<std::complex<double>>, "Convert to double complex tensor")
            .def("norm_max", &T::template norm<-1>, "Get -1 norm, namely max absolute value")
            .def("norm_num", &T::template norm<0>, "Get 0 norm, namely number of element, note: not check whether equal to 0")
            .def("norm_1", &T::template norm<1>, "Get 1 norm, namely summation of all element absolute value")
            .def("norm_2", &T::template norm<2>, "Get 2 norm")
            .def("edge_rename", &T::edge_rename, py::arg("name_dictionary"), "Rename names of edges, which will not copy data")
            .def(
                  "transpose",
                  [](const T& tensor, const std::vector<Name>& names) { return tensor.transpose(names); },
                  py::arg("new_names"),
                  "Transpose the tensor to the order of new names")
            .def("reverse_edge",
                 &T::reverse_edge,
                 py::arg("reversed_name_set"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_set") = py::set(),
                 "Reverse fermi arrow of several edge")
            .def("merge_edge",
                 &T::merge_edge,
                 py::arg("merge_map"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_merge_set") = py::set(),
                 py::arg("parity_exclude_name_reverse_set") = py::set(),
                 "Merge several edges of the tensor into ones")
            .def("split_edge",
                 &T::split_edge,
                 py::arg("split_map"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_split_set") = py::set(),
                 "Split edges of a tensor to many edges")
            .def(
                  "contract",
                  [](const T& tensor_1, const T& tensor_2, std::set<std::tuple<Name, Name>> contract_names) {
                     return T::contract(tensor_1, tensor_2, contract_names);
                  },
                  py::arg("another_tensor"),
                  py::arg("contract_names"),
                  "Contract two tensors")
            .def(
                  "contract_all_edge", [](const T& tensor) { return tensor.contract_all_edge(); }, "Contract all edge with conjugate tensor")
            .def(
                  "contract_all_edge",
                  [](const T& tensor, const T& other) { return tensor.contract_all_edge(other); },
                  py::arg("another_tensor"),
                  "Contract as much as possible with another tensor on same name edges")
            .def("conjugate", &T::conjugate, "Get the conjugate Tensor")
            .def("trace", &T::trace) // TODO trace
            .def(
                  "svd",
                  [](const T& tensor, const std::set<Name>& free_name_set_u, Name common_name_u, Name common_name_v, Size cut) {
                     auto result = tensor.svd(free_name_set_u, common_name_u, common_name_v, cut);
                     return py::make_tuple(std::move(result.U), std::move(result.S), std::move(result.V));
                  },
                  py::arg("free_name_set_u"),
                  py::arg("common_name_u"),
                  py::arg("common_name_v"),
                  py::arg("cut") = -1,
                  "Singular value decomposition")
            .def(
                  "multiple",
                  [](T& tensor, const typename T::Singular& S, const Name& name, char direction) -> T& {
                     switch (direction) {
                        case ('u'):
                        case ('U'):
                           return tensor.multiple(S, name, false);
                        case ('v'):
                        case ('V'):
                           return tensor.multiple(S, name, true);
                        default:
                           warning_or_error("Error direction in multiple");
                           return tensor;
                     }
                  },
                  py::arg("singular"),
                  py::arg("name"),
                  py::arg("direction"),
                  "Multiple with singular generated by svd",
                  py::return_value_policy::reference_internal);
   }

   template<class Symmetry, class Element, bool IsTuple>
   auto declare_edge(py::module& m, const char* name) {
      auto result = py::class_<Edge<Symmetry>>(m, name, ("Edge with symmetry type as " + std::string(name) + "Symmetry").c_str())
                          .def_readwrite("map", &Edge<Symmetry>::map)
                          .def(implicit_init<Edge<Symmetry>, Size>(), py::arg("dimension"), "Edge with only one symmetry")
                          .def(implicit_init<Edge<Symmetry>, std::map<Symmetry, Size>>(),
                               py::arg("dictionary_from_symmetry_to_dimension"),
                               "Create Edge with dictionary from symmetry to dimension")
                          .def(implicit_init<Edge<Symmetry>, const std::set<Symmetry>&>(),
                               py::arg("set_of_symmetry"),
                               "Edge with several symmetries which dimensions are all one")
                          .def("__str__",
                               [](const Edge<Symmetry>& edge) {
                                  auto out = std::stringstream();
                                  out << edge;
                                  return out.str();
                               })
                          .def("__repr__", [](const Edge<Symmetry>& edge) {
                             auto out = std::stringstream();
                             out << "Edge";
                             if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
                                out << "[";
                             }
                             out << edge;
                             if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
                                out << "]";
                             }
                             return out.str();
                          });
      if constexpr (!std::is_same_v<Element, void>) {
         // is not no symmetry
         result = result.def(implicit_init<Edge<Symmetry>, std::map<Element, Size>>([](const std::map<Element, Size>& element_map) {
                                auto symmetry_map = std::map<Symmetry, Size>();
                                for (const auto& [key, value] : element_map) {
                                   if constexpr (IsTuple) {
                                      symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                   } else {
                                      symmetry_map[Symmetry(key)] = value;
                                   }
                                }
                                return Edge<Symmetry>(std::move(symmetry_map));
                             }),
                             py::arg("dictionary_from_symmetry_to_dimension"),
                             "Create Edge with dictionary from symmetry to dimension")
                        .def(implicit_init<Edge<Symmetry>, const std::set<Element>&>([](const std::set<Element>& element_set) {
                                auto symmetry_set = std::set<Symmetry>();
                                for (const auto& element : element_set) {
                                   if constexpr (IsTuple) {
                                      symmetry_set.insert(std::make_from_tuple<Symmetry>(element));
                                   } else {
                                      symmetry_set.insert(Symmetry(element));
                                   }
                                }
                                return Edge<Symmetry>(std::move(symmetry_set));
                             }),
                             py::arg("set_of_symmetry"),
                             "Edge with several symmetries which dimensions are all one");
      }
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         // is fermi symmetry
         result = result.def_readwrite("arrow", &Edge<Symmetry>::arrow, "Fermi Arrow")
                        .def(py::init<Arrow, std::map<Symmetry, Size>>(),
                             py::arg("arrow"),
                             py::arg("dictionary_from_symmetry_to_dimension"),
                             "Fermi Edge created from arrow and dictionary")
                        .def(implicit_init<Edge<Symmetry>, std::tuple<Arrow, std::map<Symmetry, Size>>>(
                                   [](const std::tuple<Arrow, std::map<Symmetry, Size>>& p) { return std::make_from_tuple<Edge<Symmetry>>(p); }),
                             py::arg("tuple_of_arrow_and_dictionary"),
                             "Fermi Edge created from arrow and dictionary");
         if constexpr (!std::is_same_v<Element, void>) {
            // always true
            result = result.def(py::init([](Arrow arrow, const std::map<Element, Size>& element_map) {
                                   auto symmetry_map = std::map<Symmetry, Size>();
                                   for (const auto& [key, value] : element_map) {
                                      if constexpr (IsTuple) {
                                         symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                      } else {
                                         symmetry_map[Symmetry(key)] = value;
                                      }
                                   }
                                   return Edge<Symmetry>(arrow, symmetry_map);
                                }),
                                py::arg("arrow"),
                                py::arg("dictionary_from_symmetry_to_dimension"),
                                "Fermi Edge created from arrow and dictionary")
                           .def(implicit_init<Edge<Symmetry>, std::tuple<Arrow, std::map<Element, Size>>>(
                                      [](const std::tuple<Arrow, std::map<Element, Size>>& p) {
                                         const auto& [arrow, element_map] = p;
                                         auto symmetry_map = std::map<Symmetry, Size>();
                                         for (const auto& [key, value] : element_map) {
                                            if constexpr (IsTuple) {
                                               symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                            } else {
                                               symmetry_map[Symmetry(key)] = value;
                                            }
                                         }
                                         return Edge<Symmetry>(arrow, symmetry_map);
                                      }),
                                py::arg("tuple_of_arrow_and_dictionary"),
                                "Fermi Edge created from arrow and dictionary");
         }
      }
      return result;
   }

   template<class Symmetry>
   auto declare_symmetry(py::module& m, const char* name) {
      return py::class_<Symmetry>(m, name, (std::string(name) + "Symmetry").c_str())
            .def("__repr__",
                 [=](const Symmetry& symmetry) {
                    auto out = std::stringstream();
                    out << name;
                    out << "Symmetry";
                    out << "[";
                    out << symmetry;
                    out << "]";
                    return out.str();
                 })
            .def("__str__", [=](const Symmetry& symmetry) {
               auto out = std::stringstream();
               out << symmetry;
               return out.str();
            });
   }

   PYBIND11_MODULE(TAT, m) {
      m.doc() = "TAT is A Tensor library!";
      m.attr("version") = version;
      // name
      py::class_<Name>(m, "Name", "Name used in edge of tensor, which is just a string but stored by identical integer")
            .def(implicit_init<Name, const std::string&>(), py::arg("name"), "Name with specified name")
            .def(py::init<int>(), py::arg("id"), "Name with specified id directly")
            .def("__repr__", [](const Name& name) { return "Name[" + id_to_name.at(name.id) + "]"; })
            .def("__str__", [](const Name& name) { return id_to_name.at(name.id); })
            .def_readonly("id", &Name::id)
            .def_property_readonly("name", [](const Name& name) { return id_to_name.at(name.id); });

      // symmetry
      auto symmetry_m = m.def_submodule("Symmetry", "All kinds of symmetries for TAT");
      declare_symmetry<NoSymmetry>(symmetry_m, "No").def(py::init<>());
      declare_symmetry<Z2Symmetry>(symmetry_m, "Z2").def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2")).def_readwrite("z2", &Z2Symmetry::z2);
      declare_symmetry<U1Symmetry>(symmetry_m, "U1").def(implicit_init<U1Symmetry, U1>(), py::arg("u1")).def_readwrite("u1", &U1Symmetry::u1);
      declare_symmetry<FermiSymmetry>(symmetry_m, "Fermi")
            .def(implicit_init<FermiSymmetry, Fermi>(), py::arg("fermi"))
            .def_readwrite("fermi", &FermiSymmetry::fermi);
      declare_symmetry<FermiZ2Symmetry>(symmetry_m, "FermiZ2")
            .def(py::init<Fermi, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_init<FermiZ2Symmetry, const std::tuple<Fermi, Z2>&>(
                       [](const std::tuple<Fermi, Z2>& p) { return std::make_from_tuple<FermiZ2Symmetry>(p); }),
                 py::arg("tuple_of_fermi_z2"))
            .def_readwrite("fermi", &FermiZ2Symmetry::fermi)
            .def_readwrite("z2", &FermiZ2Symmetry::z2);
      declare_symmetry<FermiU1Symmetry>(symmetry_m, "FermiU1")
            .def(py::init<Fermi, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_init<FermiU1Symmetry, const std::tuple<Fermi, U1>&>(
                       [](const std::tuple<Fermi, U1>& p) { return std::make_from_tuple<FermiU1Symmetry>(p); }),
                 py::arg("tuple_of_fermi_u1"))
            .def_readwrite("fermi", &FermiU1Symmetry::fermi)
            .def_readwrite("u1", &FermiU1Symmetry::u1);
      // edge
      auto edge_m = m.def_submodule("Edge", "Edges of all kinds of symmetries for TAT");
      declare_edge<NoSymmetry, void, false>(edge_m, "No");
      declare_edge<Z2Symmetry, Z2, false>(edge_m, "Z2");
      declare_edge<U1Symmetry, U1, false>(edge_m, "U1");
      declare_edge<FermiSymmetry, Fermi, false>(edge_m, "Fermi");
      declare_edge<FermiZ2Symmetry, std::tuple<Fermi, Z2>, true>(edge_m, "FermiZ2");
      declare_edge<FermiU1Symmetry, std::tuple<Fermi, U1>, true>(edge_m, "FermiU1");
      // tensor
      auto tensor_m = m.def_submodule("Tensor", "Tensors for TAT");
      auto singular_m = m.def_submodule("Singular", "Singulars for TAT, used in svd");
      auto block_m = m.def_submodule("Block", "Block of Tensor for TAT");
#define DECLARE_TENSOR(SCALAR, SCALARNAME, SYMMETRY) \
   declare_tensor<SCALAR, SYMMETRY##Symmetry>(tensor_m, singular_m, block_m, SCALARNAME #SYMMETRY, #SCALAR, #SYMMETRY "Symmetry");
#define DECLARE_TENSOR_WITH_SAME_SCALAR(SCALAR, SCALARNAME) \
   do {                                                     \
      DECLARE_TENSOR(SCALAR, SCALARNAME, No);               \
      DECLARE_TENSOR(SCALAR, SCALARNAME, Z2);               \
      DECLARE_TENSOR(SCALAR, SCALARNAME, U1);               \
      DECLARE_TENSOR(SCALAR, SCALARNAME, Fermi);            \
      DECLARE_TENSOR(SCALAR, SCALARNAME, FermiZ2);          \
      DECLARE_TENSOR(SCALAR, SCALARNAME, FermiU1);          \
   } while (false)
      DECLARE_TENSOR_WITH_SAME_SCALAR(float, "S");
      DECLARE_TENSOR_WITH_SAME_SCALAR(double, "D");
      DECLARE_TENSOR_WITH_SAME_SCALAR(std::complex<float>, "C");
      DECLARE_TENSOR_WITH_SAME_SCALAR(std::complex<double>, "Z");
#undef DECLARE_TENSOR_WITH_SAME_SCALAR
#undef DECLARE_TENSOR
      // mpi
#ifdef TAT_USE_MPI
      auto mpi_m = m.def_submodule("mpi", "mpi support for TAT");
      mpi_m.def("barrier", &mpi::barrier);
#define DECLARE_MPI(SCALARTYPE)                        \
   do {                                                \
      declare_mpi<SCALARTYPE, NoSymmetry>(mpi_m);      \
      declare_mpi<SCALARTYPE, Z2Symmetry>(mpi_m);      \
      declare_mpi<SCALARTYPE, U1Symmetry>(mpi_m);      \
      declare_mpi<SCALARTYPE, FermiSymmetry>(mpi_m);   \
      declare_mpi<SCALARTYPE, FermiZ2Symmetry>(mpi_m); \
      declare_mpi<SCALARTYPE, FermiU1Symmetry>(mpi_m); \
   } while (false)
      DECLARE_MPI(float);
      DECLARE_MPI(double);
      DECLARE_MPI(std::complex<float>);
      DECLARE_MPI(std::complex<double>);
#undef DECLARE_MPI
      mpi_m.attr("rank") = mpi::mpi.rank;
      mpi_m.attr("size") = mpi::mpi.size;
      mpi_m.def("print", [](py::args args, py::kwargs kwargs) {
         if (mpi::mpi.rank == 0) {
            py::print(*args, **kwargs);
         }
      });
#endif
      at_exit.release();
   }
} // namespace TAT
