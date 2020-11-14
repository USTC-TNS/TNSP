/**
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <random>
#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <TAT/TAT.hpp>

#define TAT_LOOP_ALL_SCALAR                   \
   TAT_SINGLE_SCALAR(S, float);               \
   TAT_SINGLE_SCALAR(D, double);              \
   TAT_SINGLE_SCALAR(C, std::complex<float>); \
   TAT_SINGLE_SCALAR(Z, std::complex<double>);

#define TAT_LOOP_ALL_SYMMETRY    \
   TAT_SINGLE_SYMMETRY(No);      \
   TAT_SINGLE_SYMMETRY(Z2);      \
   TAT_SINGLE_SYMMETRY(U1);      \
   TAT_SINGLE_SYMMETRY(Fermi);   \
   TAT_SINGLE_SYMMETRY(FermiZ2); \
   TAT_SINGLE_SYMMETRY(FermiU1);

#define TAT_SINGLE_SYMMETRY_ALL_SCALAR(SYM)                 \
   TAT_SINGLE_SCALAR_SYMMETRY(S, float, SYM);               \
   TAT_SINGLE_SCALAR_SYMMETRY(D, double, SYM);              \
   TAT_SINGLE_SCALAR_SYMMETRY(C, std::complex<float>, SYM); \
   TAT_SINGLE_SCALAR_SYMMETRY(Z, std::complex<double>, SYM);

#define TAT_LOOP_ALL_SCALAR_SYMMETRY        \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(No);      \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Z2);      \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(U1);      \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Fermi);   \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiZ2); \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiU1);

PYBIND11_MAKE_OPAQUE(std::vector<TAT::Name>)

#define TAT_SINGLE_SYMMETRY(SYM)                                                                \
   PYBIND11_MAKE_OPAQUE(std::map<TAT::SYM##Symmetry, TAT::Size>);                               \
   PYBIND11_MAKE_OPAQUE(std::vector<TAT::SYM##Symmetry>);                                       \
   PYBIND11_MAKE_OPAQUE(std::vector<TAT::Edge<TAT::SYM##Symmetry>>);                            \
   PYBIND11_MAKE_OPAQUE(std::vector<std::tuple<TAT::Name, TAT::BoseEdge<TAT::SYM##Symmetry>>>); \
   PYBIND11_MAKE_OPAQUE(std::map<TAT::Name, std::vector<std::tuple<TAT::Name, TAT::BoseEdge<TAT::SYM##Symmetry>>>>);
TAT_LOOP_ALL_SYMMETRY;
#undef TAT_SINGLE_SYMMETRY

#define TAT_SINGLE_SCALAR(SCALARSHORT, SCALAR) PYBIND11_MAKE_OPAQUE(TAT::vector<SCALAR>);
TAT_LOOP_ALL_SCALAR
#undef TAT_SINGLE_SCALAR

#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) PYBIND11_MAKE_OPAQUE(std::map<std::vector<TAT::SYM##Symmetry>, TAT::vector<SCALAR>>);
TAT_LOOP_ALL_SCALAR_SYMMETRY;
#undef TAT_SINGLE_SCALAR_SYMMETRY

namespace TAT {
   namespace py = pybind11;

   auto random_engine = std::default_random_engine(std::random_device()());
   void set_random_seed(unsigned long seed) {
      random_engine.seed(seed);
   }

   template<typename T>
   std::unique_ptr<T> to_unique(T&& object) {
      return std::make_unique<T>(std::move(object));
   }

   struct AtExit {
      std::vector<std::function<void()>> function_list;
      void operator()(std::function<void()>&& function) {
         function_list.push_back(std::move(function));
      }
      void release() {
         for (auto& function : function_list) {
            function();
         }
         function_list.resize(0);
      }
   };
   AtExit at_exit;

   template<typename Type, typename Args>
   auto implicit_init() {
      at_exit([]() { py::implicitly_convertible<Args, Type>(); });
      return py::init<Args>();
   }

   template<typename Type, typename Args, typename Func>
   auto implicit_init(Func&& func) {
      at_exit([]() { py::implicitly_convertible<Args, Type>(); });
      return py::init(func);
   }

   template<typename ScalarType, typename Symmetry>
   struct blocks_of_tensor {
      py::object tensor;
   };

   template<typename ScalarType, typename Symmetry>
   struct unordered_block_of_tensor {
      py::object tensor;
      std::map<Name, Symmetry> position;
   };

   template<typename ScalarType, typename Symmetry>
   struct ordered_block_of_tensor {
      py::object tensor;
      std::vector<std::tuple<Name, Symmetry>> position;
   };

   template<typename Symmetry>
   auto generate_vector_of_name_and_symmetry(const std::vector<Name>& position) {
      auto real_position = std::vector<std::tuple<Name, Symmetry>>();
      for (const auto& n : position) {
         real_position.push_back({n, Symmetry()});
      }
      return real_position;
   }

   template<typename Block>
   auto try_get_numpy_array(Block& block) {
      auto result = py::cast(block, py::return_value_policy::move);
      try {
         return py::module::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         return result;
      }
   }

   template<typename Block>
   auto try_set_numpy_array(Block& block, const py::object& object) {
      auto result = py::cast(block, py::return_value_policy::move);
      try {
         result = py::module::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         throw std::runtime_error("Cannot import numpy but setting block of tensor need numpy");
      }
      result.attr("__setitem__")(py::ellipsis(), object);
   }

   template<typename ScalarType, typename Symmetry>
   void declare_tensor(
         py::module& tensor_m,
         py::module& singular_m,
         py::module& block_m,
         py::module& edge_m,
         py::module& tat_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      using T = Tensor<ScalarType, Symmetry>;
      using S = Singular<ScalarType, Symmetry>;
      using E = Edge<Symmetry>;
      using BS = blocks_of_tensor<ScalarType, Symmetry>;
      using B1 = unordered_block_of_tensor<ScalarType, Symmetry>;
      using B2 = ordered_block_of_tensor<ScalarType, Symmetry>;
      std::string tensor_name = scalar_short_name + symmetry_short_name;
      py::class_<BS>(
            block_m,
            (tensor_name + "S").c_str(),
            ("Blocks of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def("__getitem__",
                 [](const BS& bs, std::map<Name, Symmetry> position) {
                    auto block = B1{bs.tensor, std::move(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__",
                 [](BS& bs, std::map<Name, Symmetry> position, const py::object& object) {
                    auto block = B1{bs.tensor, std::move(position)};
                    try_set_numpy_array(block, object);
                 })
            .def("__getitem__",
                 [](const BS& bs, std::vector<std::tuple<Name, Symmetry>> position) {
                    auto block = B2{bs.tensor, std::move(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__",
                 [](BS& bs, std::vector<std::tuple<Name, Symmetry>> position, const py::object& object) {
                    auto block = B2{bs.tensor, std::move(position)};
                    try_set_numpy_array(block, object);
                 })
            .def("__getitem__",
                 [](const BS& bs, const std::vector<Name>& position) {
                    auto block = B2{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__", [](BS& bs, const std::vector<Name>& position, const py::object& object) {
               auto block = B2{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)};
               try_set_numpy_array(block, object);
            });
      py::class_<B1>(
            block_m,
            ("U_" + tensor_name).c_str(),
            ("Unordered Block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](B1& b) {
               // 返回的buffer是可变的
               auto& tensor = py::cast<T&>(b.tensor);
               auto& block = tensor.block(b.position);
               const Rank rank = tensor.names.size();
               auto dimensions = std::vector<int>(rank);
               auto leadings = std::vector<int>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = tensor.core->edges[i].map.at(b.position[tensor.names[i]]);
                  // 使用operator[]在NoSymmetry时获得默认对称性, 从而得到仅有的维度
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
      py::class_<B2>(
            block_m,
            ("O_" + tensor_name).c_str(),
            ("Ordered Block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](B2& b) {
               // 返回的buffer是可变的
               auto& tensor = py::cast<T&>(b.tensor);
               auto position_map = std::map<Name, Symmetry>();
               for (const auto& [name, symmetry] : b.position) {
                  position_map[name] = symmetry;
               }
               auto& block = tensor.block(position_map);
               const Rank rank = tensor.names.size();
               auto dimensions = std::vector<int>(rank);
               auto leadings = std::vector<int>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = tensor.core->edges[i].map.at(position_map[tensor.names[i]]);
                  // 使用operator[]在NoSymmetry时获得默认对称性, 从而得到仅有的维度
               }
               for (auto i = rank; i-- > 0;) {
                  if (i == rank - 1) {
                     leadings[i] = sizeof(ScalarType);
                  } else {
                     leadings[i] = leadings[i + 1] * dimensions[i + 1];
                  }
               }
               auto real_dimensions = std::vector<int>(rank);
               auto real_leadings = std::vector<int>(rank);
               for (auto i = 0; i < rank; i++) {
                  auto j = tensor.name_to_index.at(std::get<0>(b.position[i]));
                  real_dimensions[i] = dimensions[j];
                  real_leadings[i] = leadings[j];
               }
               return py::buffer_info{
                     block.data(),
                     sizeof(ScalarType),
                     py::format_descriptor<ScalarType>::format(),
                     rank,
                     std::move(real_dimensions),
                     std::move(real_leadings)};
            });
      ScalarType one = 1;
      if constexpr (is_complex_v<ScalarType>) {
         one = ScalarType(1, 1);
      }
      py::class_<S>(
            singular_m,
            tensor_name.c_str(),
            ("Singulars in tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def_readonly("value", &S::value, "singular value dictionary")
            .def("__str__", [](const S& s) { return s.show(); })
            .def("norm_max", &S::template norm<-1>)
            .def("norm_sum", &S::template norm<1>)
            .def("dump", [](S& s) { return py::bytes(s.dump()); }) // 这个copy很难去掉, dump内一次copy, string to byte一次copy
            .def(
                  "load", [](S& s, const py::bytes& bytes) -> S& { return s.load(std::string(bytes)); }, py::return_value_policy::reference_internal)
            .def(py::pickle(
                  [](const S& s) { return py::bytes(s.dump()); }, [](const py::bytes& bytes) { return to_unique(S().load(std::string(bytes))); }))
            .def(py::self += real_base_t<ScalarType>())
            .def(py::self -= real_base_t<ScalarType>())
            .def(py::self *= real_base_t<ScalarType>())
            .def(py::self /= real_base_t<ScalarType>());
      py::class_<T>(
            tensor_m,
            tensor_name.c_str(),
            ("Tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def_readonly("name", &T::names, "Names of all edge of the tensor")
            .def_property_readonly(
                  "edge", [](const T& tensor) -> std::vector<E>& { return tensor.core->edges; }, "Edges of tensor")
            .def_property_readonly(
                  "data", [](const T& tensor) -> auto& { return tensor.core->blocks; }, "All block data of the tensor")
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
                    if (tensor.is_valid()) {
                       return tensor.show();
                    } else {
                       return std::string("{}");
                    }
                 })
            .def("__repr__",
                 [tensor_name](const T& tensor) {
                    auto out = std::stringstream();
                    out << tensor_name << "Tensor";
                    out << '{';
                    if (tensor.is_valid()) {
                       out << console_green << "names" << console_origin << ':';
                       out << tensor.names;
                       out << ',' << console_green << "edges" << console_origin << ':';
                       out << tensor.core->edges;
                    }
                    out << '}';
                    return out.str();
                 })
            .def(py::init<>(), "Default Constructor")
            .def(py::init<>([edge_m, symmetry_short_name](std::vector<Name> names, const py::list& edges, bool auto_reverse) {
                    auto tensor_edges = std::vector<E>();
                    for (const auto& this_edge : edges) {
                       tensor_edges.push_back(py::cast<E>(edge_m.attr(symmetry_short_name.c_str())(this_edge)));
                    }
                    return T(std::move(names), std::move(tensor_edges), auto_reverse);
                 }),
                 py::arg("names"),
                 py::arg("edges"),
                 py::arg("auto_reverse") = false,
                 "Construct tensor with edge names and edge shapes")
            .def(implicit_init<T, ScalarType>(), py::arg("number"), "Create rank 0 tensor with only one element")
            .def(py::init<>([](const std::string& string) {
                    auto ss = std::stringstream(string);
                    auto result = T();
                    ss >> result;
                    return result;
                 }),
                 "Read tensor from text string")
            .def_static(
                  "one",
                  &T::one,
                  py::arg("number"),
                  py::arg("names"),
                  py::arg("edge_symmetry") = py::list(),
                  py::arg("edge_arrow") = py::list(),
                  "Create tensor with high rank but containing only one element")
            .def(is_complex_v<ScalarType> ? "__complex__" : "__float__", [](const T& tensor) -> ScalarType { return tensor; })
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
                  "sqrt",
                  [](const T& tensor) { return tensor.map([](ScalarType value) { return std::sqrt(value); }); },
                  "Get elementwise square root")
            .def(
                  "set",
                  [](T& tensor, std::function<ScalarType()>& function) -> T& { return tensor.set(function); },
                  py::arg("function"),
                  "Set every element of a tensor by a function",
                  py::return_value_policy::reference_internal)
            .def(
                  "zero", [](T& tensor) -> T& { return tensor.zero(); }, "Set all element zero", py::return_value_policy::reference_internal)
            .def(
                  "identity",
                  [](T& tensor, const std::set<std::tuple<Name, Name>>& pair) -> T& { return tensor.identity(pair); },
                  py::arg("pairs"),
                  "Set tensor to identity like matrix",
                  py::return_value_policy::reference_internal)
            .def(
                  "test",
                  [](T& tensor, ScalarType first, ScalarType step) -> T& { return tensor.test(first, step); },
                  py::arg("first") = 0,
                  py::arg("step") = 1,
                  "Useful function generate simple data in tensor element for test",
                  py::return_value_policy::reference_internal)
            .def_property_readonly("block", [](const py::object& tensor) { return blocks_of_tensor<ScalarType, Symmetry>{tensor}; })
            .def(
                  "__getitem__",
                  [](const T& tensor, const std::map<Name, typename T::EdgeInfoForGetItem>& position) { return tensor.at(position); },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"))
            .def(
                  "__setitem__",
                  [](T& tensor, const std::map<Name, typename T::EdgeInfoForGetItem>& position, const ScalarType& value) {
                     tensor.at(position) = value;
                  },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"),
                  py::arg("value"))
            .def("slice", &T::slice, "Get Slice of tensor", py::arg("configure"), py::arg("new_name") = "Null", py::arg("arrow") = false)
            .def("expand", &T::expand, "Expand Edge of tensor", py::arg("configure"), py::arg("old_name") = "Null")
            .def(
                  "to",
                  [](const T& tensor, const py::object& object) -> py::object {
                     auto string = py::str(object);
                     auto contain = [&string](const char* other) { return py::cast<bool>(string.attr("__contains__")(other)); };
                     if (contain("float32")) {
                        return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                     } else if (contain("complex32")) {
                        return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                     } else if (contain("float")) {
                        return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                     } else if (contain("complex")) {
                        return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                     } else if (contain("S")) {
                        return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                     } else if (contain("D")) {
                        return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                     } else if (contain("C")) {
                        return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                     } else if (contain("Z")) {
                        return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                     } else {
                        throw std::runtime_error("Invalid scalar type in type conversion");
                     }
                  },
                  "Convert to other scalar type tensor")
            .def("norm_max", &T::template norm<-1>, "Get -1 norm, namely max absolute value")
            .def("norm_num", &T::template norm<0>, "Get 0 norm, namely number of element, note: not check whether equal to 0")
            .def("norm_sum", &T::template norm<1>, "Get 1 norm, namely summation of all element absolute value")
            .def("norm_2", &T::template norm<2>, "Get 2 norm")
            .def("edge_rename", &T::edge_rename, py::arg("name_dictionary"), "Rename names of edges, which will not copy data")
            .def(
                  "transpose",
                  [](const T& tensor, std::vector<Name> names) { return tensor.transpose(std::move(names)); },
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
            .def(
                  "split_edge",
                  [](const T& tensor, const py::dict& split, const bool apply_parity, const std::set<Name>& parity_exclude_name_split) {
                     auto tensor_split = std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>();
                     for (const auto& [name, named_edge_list] : split) {
                        tensor_split[py::cast<Name>(name)] = py::cast<std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>(named_edge_list);
                     }
                     return tensor.split_edge(std::move(tensor_split), apply_parity, parity_exclude_name_split);
                  },
                  py::arg("split_map"),
                  py::arg("apply_parity") = false,
                  py::arg("parity_exclude_name_split_set") = py::set(),
                  "Split edges of a tensor to many edges")
            .def(
                  "edge_operator",
                  [](const T& tensor,
                     const std::map<Name, Name>& rename_map,
                     const py::dict& split,
                     const std::set<Name>& reversed_name,
                     const std::map<Name, std::vector<Name>>& merge_map,
                     std::vector<Name> new_names,
                     const bool apply_parity,
                     std::set<Name> parity_exclude_name_split_set,
                     std::set<Name> parity_exclude_name_reverse_set,
                     std::set<Name> parity_exclude_name_reverse_before_merge_set,
                     std::set<Name> parity_exclude_name_merge_set,
                     const std::map<Name, std::map<Symmetry, Size>>& edge_and_symmetries_to_cut_before_all = {}) {
                     auto split_map = std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>();
                     for (const auto& [name, named_edge_list] : split) {
                        split_map[py::cast<Name>(name)] = py::cast<std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>(named_edge_list);
                     }
                     return tensor.edge_operator(
                           rename_map,
                           split_map,
                           reversed_name,
                           merge_map,
                           std::move(new_names),
                           apply_parity,
                           {std::move(parity_exclude_name_split_set),
                            std::move(parity_exclude_name_reverse_set),
                            std::move(parity_exclude_name_reverse_before_merge_set),
                            std::move(parity_exclude_name_merge_set)},
                           edge_and_symmetries_to_cut_before_all);
                  },
                  py::arg("rename_map"),
                  py::arg("split_map"),
                  py::arg("reversed_name"),
                  py::arg("merge_map"),
                  py::arg("new_names"),
                  py::arg("apply_parity") = false,
                  py::arg("parity_exclude_name_split_set") = py::set(),
                  py::arg("parity_exclude_name_reverse_set") = py::set(),
                  py::arg("parity_exclude_name_reverse_before_merge_set") = py::set(),
                  py::arg("parity_exclude_name_merge_set") = py::set(),
                  py::arg("edge_and_symmetries_to_cut_before_all") = py::dict(),
                  "Tensor Edge Operator")
            .def(
                  "contract",
                  [](const T& tensor_1, const T& tensor_2, std::set<std::tuple<Name, Name>> contract_names) {
                     return T::contract(tensor_1, tensor_2, std::move(contract_names));
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
            .def("exponential", &T::exponential, py::arg("pairs"), py::arg("step") = 2, "Calculate exponential like matrix")
            .def("conjugate", &T::conjugate, "Get the conjugate Tensor")
            .def("trace", &T::trace)
            .def(
                  "svd",
                  [](const T& tensor, const std::set<Name>& free_name_set_u, const Name& common_name_u, const Name& common_name_v, Size cut) {
                     auto result = tensor.svd(free_name_set_u, common_name_u, common_name_v, cut);
                     return py::make_tuple(std::move(result.U), std::move(result.S), std::move(result.V));
                  },
                  py::arg("free_name_set_u"),
                  py::arg("common_name_u"),
                  py::arg("common_name_v"),
                  py::arg("cut") = Size(-1),
                  "Singular value decomposition")
            .def(
                  "qr",
                  [](const T& tensor,
                     char free_name_direction,
                     const std::set<Name>& free_name_set,
                     const Name& common_name_q,
                     const Name& common_name_r) {
                     auto result = tensor.qr(free_name_direction, free_name_set, common_name_q, common_name_r);
                     return py::make_tuple(std::move(result.Q), std::move(result.R));
                  },
                  py::arg("free_name_direction"),
                  py::arg("free_name_set"),
                  py::arg("common_name_q"),
                  py::arg("common_name_r"),
                  "QR decomposition")
            .def(
                  "multiple",
                  [](T& tensor, const typename T::SingularType& s, const Name& name, char direction, bool division) {
                     return tensor.multiple(s, name, direction, division);
                  },
                  py::arg("singular"),
                  py::arg("name"),
                  py::arg("direction"),
                  py::arg("division") = false,
                  "Multiple with singular generated by svd")
            .def(
                  "dump", [](T& tensor) { return py::bytes(tensor.dump()); }, "dump Tensor to bytes")
            .def(
                  "load",
                  [](T& tensor, const py::bytes& bytes) -> T& { return tensor.load(std::string(bytes)); },
                  "Load Tensor from bytes",
                  py::return_value_policy::reference_internal)
            .def(py::pickle(
                  [](const T& tensor) { return py::bytes(tensor.dump()); },
                  // 这里必须是make_unique, 很奇怪, 可能是pybind11的bug
                  [](const py::bytes& bytes) { return to_unique(T().load(std::string(bytes))); }))
#ifdef TAT_USE_MPI
            .def("send", &T::send, py::arg("destination"), "Send a tensor to destination")
            .def_static("receive", &T::receive, py::arg("source"), "Receive a tensor from source") // static ?
            .def("send_receive", &T::send_receive, py::arg("source"), py::arg("destination"), "Send tensor from source to destination")
            .def("broadcast", &T::broadcast, py::arg("root"), "Broadcast a tensor from root")
            .def(
                  "reduce",
                  [](const T& tensor, const int root, std::function<T(T, T)>& function) { return tensor.reduce(root, function); },
                  py::arg("root"),
                  py::arg("function"),
                  "Reduce a tensor with commutative function into root")
            .def("summary", &T::summary, py::arg("root"), "Summation of a tensor into root")
            .def_static("barrier", &T::barrier, "MPI barrier")
            .def_readonly_static("mpi", &T::mpi, "MPI Handle")
#endif
            .def_static("set_random_seed", &set_random_seed, "Set Random Seed")
            .def(
                  "rand",
                  [](T& tensor, ScalarType min, ScalarType max) -> T& {
                     if constexpr (is_complex_v<ScalarType>) {
                        auto distribution_real = std::uniform_real_distribution<real_base_t<ScalarType>>(min.real(), max.real());
                        auto distribution_imag = std::uniform_real_distribution<real_base_t<ScalarType>>(min.imag(), max.imag());
                        return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                           return {distribution_real(random_engine), distribution_imag(random_engine)};
                        });
                     } else {
                        auto distribution = std::uniform_real_distribution<real_base_t<ScalarType>>(min, max);
                        return tensor.set([&distribution]() { return distribution(random_engine); });
                     }
                  },
                  py::arg("min") = 0,
                  py::arg("max") = one,
                  "Set Uniform Random Number into Tensor",
                  py::return_value_policy::reference_internal)
            .def(
                  "randn",
                  [](T& tensor, ScalarType mean, ScalarType stddev) -> T& {
                     if constexpr (is_complex_v<ScalarType>) {
                        auto distribution_real = std::normal_distribution<real_base_t<ScalarType>>(mean.real(), stddev.real());
                        auto distribution_imag = std::normal_distribution<real_base_t<ScalarType>>(mean.imag(), stddev.imag());
                        return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                           return {distribution_real(random_engine), distribution_imag(random_engine)};
                        });
                     } else {
                        auto distribution = std::normal_distribution<real_base_t<ScalarType>>(mean, stddev);
                        return tensor.set([&distribution]() { return distribution(random_engine); });
                     }
                  },
                  py::arg("mean") = 0,
                  py::arg("stddev") = one,
                  "Set Normal Distribution Random Number into Tensor",
                  py::return_value_policy::reference_internal)
            .def_readonly_static("version", &version)
            .def_readonly_static("information", &information)
            .def_readonly_static("mpi_enabled", &mpi_enabled);
   }

   template<typename Symmetry, typename Element, bool IsTuple, template<typename, bool = false> class EdgeType = Edge>
   auto declare_edge(py::module& edge_m, const char* name) {
      auto result = py::class_<EdgeType<Symmetry>>(edge_m, name, ("Edge with symmetry type as " + std::string(name) + "Symmetry").c_str())
                          .def_readonly("map", &EdgeType<Symmetry>::map)
                          .def(implicit_init<EdgeType<Symmetry>, Size>(), py::arg("dimension"), "Edge with only one symmetry")
                          .def(implicit_init<EdgeType<Symmetry>, std::map<Symmetry, Size>>(),
                               py::arg("dictionary_from_symmetry_to_dimension"),
                               "Create Edge with dictionary from symmetry to dimension")
                          .def(implicit_init<EdgeType<Symmetry>, const std::set<Symmetry>&>(),
                               py::arg("set_of_symmetry"),
                               "Edge with several symmetries which dimensions are all one");
      if constexpr (is_edge_v<EdgeType<Symmetry>>) {
         result = result.def("__str__",
                             [](const EdgeType<Symmetry>& edge) {
                                auto out = std::stringstream();
                                out << edge;
                                return out.str();
                             })
                        .def("__repr__", [name](const EdgeType<Symmetry>& edge) {
                           auto out = std::stringstream();
                           out << name << "Edge";
                           if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
                              out << "[";
                           }
                           out << edge;
                           if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
                              out << "]";
                           }
                           return out.str();
                        });
      }
      if constexpr (!std::is_same_v<Element, void>) {
         // is not no symmetry
         result = result.def(implicit_init<EdgeType<Symmetry>, std::map<Element, Size>>([](const std::map<Element, Size>& element_map) {
                                auto symmetry_map = std::map<Symmetry, Size>();
                                for (const auto& [key, value] : element_map) {
                                   if constexpr (IsTuple) {
                                      symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                   } else {
                                      symmetry_map[Symmetry(key)] = value;
                                   }
                                }
                                return EdgeType<Symmetry>(std::move(symmetry_map));
                             }),
                             py::arg("dictionary_from_symmetry_to_dimension"),
                             "Create Edge with dictionary from symmetry to dimension")
                        .def(implicit_init<EdgeType<Symmetry>, const std::set<Element>&>([](const std::set<Element>& element_set) {
                                auto symmetry_set = std::set<Symmetry>();
                                for (const auto& element : element_set) {
                                   if constexpr (IsTuple) {
                                      symmetry_set.insert(std::make_from_tuple<Symmetry>(element));
                                   } else {
                                      symmetry_set.insert(Symmetry(element));
                                   }
                                }
                                return EdgeType<Symmetry>(std::move(symmetry_set));
                             }),
                             py::arg("set_of_symmetry"),
                             "Edge with several symmetries which dimensions are all one");
      }
      if constexpr (is_fermi_symmetry_v<Symmetry> && is_edge_v<EdgeType<Symmetry>>) {
         // is fermi symmetry 且没有强制设置为BoseEdge
         result =
               result.def_readonly("arrow", &EdgeType<Symmetry>::arrow, "Fermi Arrow of the edge")
                     .def(py::init<Arrow, std::map<Symmetry, Size>>(),
                          py::arg("arrow"),
                          py::arg("dictionary_from_symmetry_to_dimension"),
                          "Fermi Edge created from arrow and dictionary")
                     .def(implicit_init<EdgeType<Symmetry>, std::tuple<Arrow, std::map<Symmetry, Size>>>(
                                [](std::tuple<Arrow, std::map<Symmetry, Size>> p) { return std::make_from_tuple<EdgeType<Symmetry>>(std::move(p)); }),
                          py::arg("tuple_of_arrow_and_dictionary"),
                          "Fermi Edge created from arrow and dictionary");
         if constexpr (!std::is_same_v<Element, void>) {
            // true if not FermiSymmetry
            result = result.def(py::init([](Arrow arrow, const std::map<Element, Size>& element_map) {
                                   auto symmetry_map = std::map<Symmetry, Size>();
                                   for (const auto& [key, value] : element_map) {
                                      if constexpr (IsTuple) {
                                         symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                      } else {
                                         symmetry_map[Symmetry(key)] = value;
                                      }
                                   }
                                   return EdgeType<Symmetry>(arrow, std::move(symmetry_map));
                                }),
                                py::arg("arrow"),
                                py::arg("dictionary_from_symmetry_to_dimension"),
                                "Fermi Edge created from arrow and dictionary")
                           .def(implicit_init<EdgeType<Symmetry>, std::tuple<Arrow, std::map<Element, Size>>>(
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
                                         return EdgeType<Symmetry>(arrow, std::move(symmetry_map));
                                      }),
                                py::arg("tuple_of_arrow_and_dictionary"),
                                "Fermi Edge created from arrow and dictionary");
         }
      }
      return result;
   }

   template<typename Symmetry>
   auto declare_symmetry(py::module& symmetry_m, const char* name) {
      return py::class_<Symmetry>(symmetry_m, name, (std::string(name) + "Symmetry").c_str())
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(-py::self)
            .def("__hash__", [](const Symmetry& symmetry) { return py::hash(py::cast(symmetry.information())); })
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

#ifdef TAT_PYTHON_MODULE
   PYBIND11_MODULE(TAT_PYTHON_MODULE, tat_m) {
#else
   PYBIND11_MODULE(TAT, tat_m) {
#endif
      tat_m.doc() = "TAT is A Tensor library!";
      tat_m.attr("version") = version;
      tat_m.attr("information") = information;
      tat_m.attr("mpi_enabled") = mpi_enabled;
      auto internal_m = tat_m.def_submodule("_internal", "internal information of TAT");
      // random
      tat_m.def("set_random_seed", &set_random_seed, "Set Random Seed");
      // mpi
#ifdef TAT_USE_MPI
      py::class_<mpi_t>(internal_m, "mpi_t", "several functions for MPI")
            .def_readonly("rank", &mpi_t::rank)
            .def_readonly("size", &mpi_t::size)
            .def_static("barrier", &mpi_t::barrier)
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
            .def("print", [](const mpi_t& mpi, const py::args& args, const py::kwargs& kwargs) {
               if (mpi.rank == 0) {
                  py::print(*args, **kwargs);
               }
            });
      tat_m.attr("mpi") = mpi;
#else
      auto mpi_fake_m = mpi_m.def_submodule("mpi", "mpi support for TAT");
      mpi_fake_m.def("print", [](const py::args& args, const py::kwargs& kwargs) { py::print(*args, **kwargs); });
#endif // MPI
      // name
      py::class_<Name>(internal_m, "Name", "Name used in edge of tensor, which is just a string but stored by identical integer")
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
#ifdef TAT_USE_SIMPLE_NAME
            .def("__hash__", [](const Name& name) { return py::hash(py::cast(name.name)); })
#else
            .def(py::init<NameIdType>(), py::arg("id"), "Name with specified id directly")
            .def_readonly("id", &Name::id)
            .def("__hash__", [](const Name& name) { return py::hash(py::cast(name.id)); })
#endif
            .def_property_readonly("name", [](const Name& name) { return name.get_name(); })
            .def(implicit_init<Name, const char*>(), py::arg("name"), "Name with specified name")
            .def("__repr__", [](const Name& name) { return "Name[" + name.get_name() + "]"; })
            .def("__str__", [](const Name& name) { return name.get_name(); });
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
      auto singular_m = internal_m.def_submodule("Singular", "Singulars for TAT, used in svd");
      auto block_m = internal_m.def_submodule("Block", "Block of Tensor for TAT");
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
   declare_tensor<SCALAR, SYM##Symmetry>(tensor_m, singular_m, block_m, edge_m, tat_m, #SCALARSHORT, #SCALAR, #SYM);
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      // stl
      auto stl_m = internal_m.def_submodule("stl", "STL bindings");
      py::bind_vector<std::vector<Name>>(stl_m, "NameList");
      py::implicitly_convertible<py::list, std::vector<Name>>();
#define TAT_SINGLE_SYMMETRY(SYM)                                                                                         \
   py::bind_map<std::map<SYM##Symmetry, Size>>(stl_m, #SYM "SymmetryEdgeMap");                                           \
   py::implicitly_convertible<py::dict, std::map<SYM##Symmetry, Size>>();                                                \
   py::bind_vector<std::vector<SYM##Symmetry>>(stl_m, #SYM "SymmetryList");                                              \
   py::implicitly_convertible<py::list, std::vector<SYM##Symmetry>>();                                                   \
   py::bind_vector<std::vector<Edge<SYM##Symmetry>>>(stl_m, #SYM "SymmetryEdgeList");                                    \
   py::implicitly_convertible<py::list, std::vector<Edge<SYM##Symmetry>>>();                                             \
   py::bind_vector<std::vector<std::tuple<Name, BoseEdge<SYM##Symmetry>>>>(stl_m, #SYM "SymmetryEdgeWithNameList");      \
   py::implicitly_convertible<py::list, std::vector<std::tuple<TAT::Name, TAT::BoseEdge<TAT::SYM##Symmetry>>>>();        \
   py::bind_map<std::map<Name, std::vector<std::tuple<Name, BoseEdge<SYM##Symmetry>>>>>(stl_m, #SYM "SymmetrySplitMap"); \
   py::implicitly_convertible<py::dict, std::map<Name, std::vector<std::tuple<Name, BoseEdge<SYM##Symmetry>>>>>();
      TAT_LOOP_ALL_SYMMETRY
#undef TAT_SINGLE_SYMMETRY
#define TAT_SINGLE_SCALAR(SCALARSHORT, SCALAR)                  \
   py::bind_vector<vector<SCALAR>>(stl_m, #SCALARSHORT "Data"); \
   py::implicitly_convertible<py::list, vector<SCALAR>>();
      TAT_LOOP_ALL_SCALAR
#undef TAT_SINGLE_SCALAR
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM)                                                               \
   py::bind_map<std::map<std::vector<TAT::SYM##Symmetry>, TAT::vector<SCALAR>>>(stl_m, #SCALARSHORT #SYM "TensorDataMap"); \
   py::implicitly_convertible<py::dict, std::map<std::vector<TAT::SYM##Symmetry>, TAT::vector<SCALAR>>>();
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      // get tensor
      tat_m.def("TAT", [tensor_m, tat_m](const py::args& args, const py::kwargs& kwargs) -> py::object {
         if (py::len(args) == 0 && py::len(kwargs) == 0) {
            std::string date = __DATE__;
            std::string year = date.substr(date.size() - 4, 4);
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
      at_exit.release();
   }
} // namespace TAT

#undef TAT_LOOP_ALL_SCALAR
#undef TAT_LOOP_ALL_SYMMETRY
#undef TAT_SINGLE_SYMMETRY_ALL_SCALAR
#undef TAT_LOOP_ALL_SCALAR_SYMMETRY
