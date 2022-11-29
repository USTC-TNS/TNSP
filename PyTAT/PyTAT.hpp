/**
 * Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef PYTAT_HPP
#define PYTAT_HPP

#include <random>
#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Do not use fast name here, because it would need frequent conversion between python str to FastName,
// which is much more expensive than the benifit of FastName with the original std::string
#define TAT_ERROR_BITS 1
#include <TAT/TAT.hpp>

#define TAT_SINGLE_SYMMETRY_ALL_SCALAR(SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(S, float, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(D, double, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(C, std::complex<float>, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(Z, std::complex<double>, SYM)

#define TAT_LOOP_ALL_SCALAR_SYMMETRY \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(No) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Z2) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(U1) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Fermi) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiZ2) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiU1) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Parity)

namespace TAT {
   namespace py = pybind11;

   inline auto random_engine = std::default_random_engine(std::random_device()());
   inline void set_random_seed(unsigned int seed) {
      random_engine.seed(seed);
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
         function_list.clear();
      }
   };
   inline AtExit at_exit;

   template<typename Type, typename Args>
   auto implicit_init() {
      at_exit([]() {
         py::implicitly_convertible<Args, Type>();
      });
      return py::init<Args>();
   }

   template<typename Type, typename Args, typename Func>
   auto implicit_init(Func&& func) {
      at_exit([]() {
         py::implicitly_convertible<Args, Type>();
      });
      return py::init(func);
   }

   // block misc

   template<typename ScalarType, typename Symmetry>
   struct storage_of_tensor {
      py::object tensor;
   };

   template<typename ScalarType, typename Symmetry>
   struct blocks_of_tensor {
      py::object tensor;
   };

   template<typename ScalarType, typename Symmetry>
   struct single_block_of_tensor {
      py::object tensor;
      std::vector<std::pair<DefaultName, Symmetry>> position;
   };

   template<typename Symmetry>
   auto generate_vector_of_name_and_symmetry(const std::vector<DefaultName>& position) {
      // used for no symmetry tensor
      auto real_position = std::vector<std::pair<DefaultName, Symmetry>>();
      for (const auto& n : position) {
         real_position.push_back({n, Symmetry()});
      }
      return real_position;
   }

   template<typename Block>
   auto try_get_numpy_array(Block&& block) {
      auto result = py::cast(std::move(block), py::return_value_policy::move); // it cast to Single Block
      try {
         return py::module_::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         return result;
      }
   }

   template<typename Block>
   auto try_set_numpy_array(Block&& block, const py::object& object) {
      auto result = py::cast(std::move(block), py::return_value_policy::move);
      try {
         result = py::module_::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         throw std::runtime_error("Cannot import numpy but setting block of tensor need numpy");
      }
      result.attr("__setitem__")(py::ellipsis(), object);
   }

   template<typename ScalarType, typename Symmetry>
   auto declare_tensor(
         py::module_& symmetry_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      auto self_m = symmetry_m.def_submodule(scalar_short_name.c_str());
      auto block_m = self_m.def_submodule("Block");
      using T = Tensor<ScalarType, Symmetry>;
      using E = Edge<Symmetry>;
      using Storage = storage_of_tensor<ScalarType, Symmetry>;
      using BS = blocks_of_tensor<ScalarType, Symmetry>;
      using B = single_block_of_tensor<ScalarType, Symmetry>;
      std::string tensor_name = scalar_short_name + symmetry_short_name;
      py::class_<Storage>(
            block_m,
            "Storage",
            ("Storage of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](Storage& storage) {
               auto& tensor = py::cast<T&>(storage.tensor);
               auto& s = tensor.storage();
               return py::buffer_info{
                     s.data(),
                     sizeof(ScalarType),
                     py::format_descriptor<ScalarType>::format(),
                     1,
                     std::vector<Size>{s.size()},
                     std::vector<Size>{sizeof(ScalarType)}};
            });
      py::class_<BS>(
            block_m,
            "Blocks",
            ("Blocks of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def("__getitem__",
                 [](const BS& bs, std::vector<std::pair<DefaultName, Symmetry>> position) {
                    return try_get_numpy_array(B{bs.tensor, std::move(position)});
                 })
            .def("__setitem__",
                 [](BS& bs, std::vector<std::pair<DefaultName, Symmetry>> position, const py::object& object) {
                    try_set_numpy_array(B{bs.tensor, std::move(position)}, object);
                 })
            .def("__getitem__",
                 [](const BS& bs, const std::vector<DefaultName>& position) {
                    return try_get_numpy_array(B{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)});
                 })
            .def("__setitem__", [](BS& bs, const std::vector<DefaultName>& position, const py::object& object) {
               try_set_numpy_array(B{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)}, object);
            });
      py::class_<B>(
            block_m,
            "Block",
            ("Single block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](B& b) {
               // return a buffer which is writable
               auto& tensor = py::cast<T&>(b.tensor);
               auto position_map = std::unordered_map<DefaultName, Symmetry>();
               for (const auto& [name, symmetry] : b.position) {
                  position_map[name] = symmetry;
               }
               auto& block = tensor.blocks(position_map);
               const Rank rank = tensor.rank();
               auto dimensions = std::vector<Size>(rank);
               auto leadings = std::vector<Size>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = tensor.edges(i).dimension_by_symmetry(position_map[tensor.names(i)]);
               }
               for (auto i = rank; i-- > 0;) {
                  if (i == rank - 1) {
                     leadings[i] = sizeof(ScalarType);
                  } else {
                     leadings[i] = leadings[i + 1] * dimensions[i + 1];
                  }
               }
               // transpose views
               auto real_dimensions = std::vector<Size>(rank);
               auto real_leadings = std::vector<Size>(rank);
               for (auto i = 0; i < rank; i++) {
                  auto j = tensor.rank_by_name(b.position[i].first);
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
      // block done

      // one is used in default random distribution
      ScalarType one = 1;
      if constexpr (is_complex<ScalarType>) {
         one = ScalarType(1, 1);
      }
      auto tensor_t = py::class_<T>(
            self_m,
            "Tensor",
            ("Tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str());
      tensor_t.attr("model") = symmetry_m;

      return [=]() mutable {
         tensor_t
               .def_property_readonly(
                     "names",
                     [](const T& tensor) {
                        return tensor.names();
                     },
                     "Names of all edge of the tensor")
               .def_property_readonly(
                     "rank",
                     [](const T& tensor) {
                        return tensor.rank();
                     })
               .def(
                     "edges",
                     [](T& tensor, Rank r) -> const E& {
                        return tensor.edges(r);
                     },
                     py::return_value_policy::reference_internal)
               .def(
                     "edges",
                     [](T& tensor, DefaultName r) -> const E& {
                        return tensor.edges(r);
                     },
                     py::return_value_policy::reference_internal)
               .def_property_readonly(
                     "blocks",
                     [](const py::object& tensor) {
                        return blocks_of_tensor<ScalarType, Symmetry>{tensor};
                     })
               .def_property(
                     "storage",
                     [](const py::object& tensor) {
                        return try_get_numpy_array(storage_of_tensor<ScalarType, Symmetry>{tensor});
                     },
                     [](const py::object& tensor, const py::object& object) {
                        return try_set_numpy_array(storage_of_tensor<ScalarType, Symmetry>{tensor}, object);
                     })
               .def(ScalarType() + py::self)
               .def(py::self + ScalarType())
               .def(py::self += ScalarType())
               .def(ScalarType() - py::self)
               .def(py::self - ScalarType())
               .def(py::self -= ScalarType())
               .def(ScalarType() * py::self)
               .def(py::self * ScalarType())
               .def(py::self *= ScalarType())
               .def(ScalarType() / py::self)
               .def(py::self / ScalarType())
               .def(py::self /= ScalarType());
#define TAT_LOOP_OPERATOR(ANOTHERSCALAR) \
   def(py::self + Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self - Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self* Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self / Tensor<ANOTHERSCALAR, Symmetry>())
         tensor_t.TAT_LOOP_OPERATOR(float).TAT_LOOP_OPERATOR(double).TAT_LOOP_OPERATOR(std::complex<float>).TAT_LOOP_OPERATOR(std::complex<double>);
#undef TAT_LOOP_OPERATOR
#define TAT_LOOP_OPERATOR(ANOTHERSCALAR) \
   def(py::self += Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self -= Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self *= Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self /= Tensor<ANOTHERSCALAR, Symmetry>())
         tensor_t.TAT_LOOP_OPERATOR(float).TAT_LOOP_OPERATOR(double);
         if constexpr (is_complex<ScalarType>) {
            tensor_t.TAT_LOOP_OPERATOR(std::complex<float>).TAT_LOOP_OPERATOR(std::complex<double>);
            tensor_t.def("__complex__", [](const T& tensor) {
               return ScalarType(tensor);
            });
         } else {
            tensor_t.def("__float__", [](const T& tensor) {
               return ScalarType(tensor);
            });
            tensor_t.def("__complex__", [](const T& tensor) {
               return std::complex<ScalarType>(ScalarType(tensor));
            });
         }
#undef TAT_LOOP_OPERATOR
         tensor_t.def_readonly_static("is_real", &is_real<ScalarType>)
               .def_readonly_static("is_complex", &is_complex<ScalarType>)
               .def("__str__",
                    [](const T& tensor) {
                       return tensor.show();
                    })
               .def("__repr__",
                    [tensor_name](const T& tensor) {
                       auto out = std::stringstream();
                       out << tensor_name << "Tensor";
                       out << tensor.shape();
                       return out.str();
                    })
               .def(py::init<>(), "Default Constructor")
               .def(py::init<std::vector<DefaultName>, std::vector<E>>(),
                    py::arg("names"),
                    py::arg("edges"),
                    "Construct tensor with edge names and edge shapes")
               .def(py::init<ScalarType, std::vector<DefaultName>, std::vector<Symmetry>, std::vector<Arrow>>(),
                    py::arg("number"),
                    py::arg("names") = py::list(),
                    py::arg("edge_symmetry") = py::list(),
                    py::arg("edge_arrow") = py::list(),
                    "Create high rank tensor with only one element")
               .def(py::init<>([](const std::string& string) {
                       auto ss = std::stringstream(string);
                       auto result = T();
                       ss >> result;
                       return result;
                    }),
                    "Read tensor from text string")
               .def("copy", &T::copy, "Deep copy a tensor")
               .def("__copy__", &T::copy)
               .def("__deepcopy__", &T::copy)
               .def("same_shape", &T::template same_shape<ScalarType>, "Create a tensor with same shape")
               .def(
                     "map",
                     [](const T& tensor, std::function<ScalarType(ScalarType)>& function) {
                        return tensor.map(function);
                     },
                     py::arg("function"),
                     "Out-place map every element of a tensor")
               .def(
                     "transform",
                     [](T& tensor, std::function<ScalarType(ScalarType)>& function) -> T& {
                        return tensor.transform(function);
                     },
                     // write function explicitly to avoid const T&/T& ambigiuous
                     // if use py::overload_cast, I need to write argument type twice
                     py::arg("function"),
                     "In-place map every element of a tensor",
                     py::return_value_policy::reference_internal)
               .def(
                     "sqrt",
                     [](const T& tensor) {
                        return tensor.map([](ScalarType value) {
                           return std::sqrt(value);
                        });
                     },
                     "Get elementwise square root") // it is faster to implement in python, since it is common used
               .def(
                     "set",
                     [](T& tensor, std::function<ScalarType()>& function) -> T& {
                        return tensor.set(function);
                     },
                     py::arg("function"),
                     "Set every element of a tensor by a function",
                     py::return_value_policy::reference_internal)
               .def(
                     "zero",
                     [](T& tensor) -> T& {
                        return tensor.zero();
                     },
                     "Set all element zero",
                     py::return_value_policy::reference_internal)
               .def(
                     "range",
                     [](T& tensor, ScalarType first, ScalarType step) -> T& {
                        return tensor.range(first, step);
                     },
                     py::arg("first") = 0,
                     py::arg("step") = 1,
                     "Useful function generate simple data in tensor element for test",
                     py::return_value_policy::reference_internal)
               .def(
                     "__getitem__",
                     [](const T& tensor, const std::unordered_map<DefaultName, std::pair<Symmetry, Size>>& position) {
                        return tensor.at(position);
                     },
                     py::arg("dictionary_from_name_to_symmetry_and_dimension"))
               .def(
                     "__getitem__",
                     [](const T& tensor, const std::unordered_map<DefaultName, Size>& position) {
                        return tensor.at(position);
                     },
                     py::arg("dictionary_from_name_to_total_index"))
               .def(
                     "__setitem__",
                     [](T& tensor, const std::unordered_map<DefaultName, std::pair<Symmetry, Size>>& position, const ScalarType& value) {
                        tensor.at(position) = value;
                     },
                     py::arg("dictionary_from_name_to_symmetry_and_dimension"),
                     py::arg("value"))
               .def(
                     "__setitem__",
                     [](T& tensor, const std::unordered_map<DefaultName, Size>& position, const ScalarType& value) {
                        tensor.at(position) = value;
                     },
                     py::arg("dictionary_from_name_to_total_index"),
                     py::arg("value"))
               .def(
                     "to",
                     [](const T& tensor, const py::object& object) -> py::object {
                        auto string = py::str(object);
                        auto contain = [&string](const char* other) {
                           return py::cast<bool>(string.attr("__contains__")(other));
                        };
                        if (contain("float32")) {
                           return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                        } else if (contain("complex64")) {
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
               .def("clear_symmetry", &T::clear_symmetry, "Convert symmetry tensor to non-symmetry tensor")
               .def(
                     "edge_rename",
                     [](const T& tensor, const std::unordered_map<DefaultName, DefaultName>& dictionary) {
                        return tensor.edge_rename(dictionary);
                     },
                     py::arg("name_dictionary"),
                     "Rename names of edges, which will not copy data")
               .def("transpose", &T::transpose, py::arg("new_names"), "Transpose the tensor to the order of new names")
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
               .def("edge_operator",
                    &T::edge_operator,
                    py::arg("split_map"),
                    py::arg("reversed_name"),
                    py::arg("merge_map"),
                    py::arg("new_names"),
                    py::arg("apply_parity") = false,
                    py::arg("parity_exclude_name_split_set") = py::set(),
                    py::arg("parity_exclude_name_reverse_before_transpose_set") = py::set(),
                    py::arg("parity_exclude_name_reverse_after_transpose_set") = py::set(),
                    py::arg("parity_exclude_name_merge_set") = py::set(),
                    "Tensor Edge Operator");
#define TAT_LOOP_CONTRACT(ANOTHERSCALAR) \
   def( \
         "contract", \
         [](const T& tensor_1, \
            const Tensor<ANOTHERSCALAR, Symmetry>& tensor_2, \
            std::unordered_set<std::pair<DefaultName, DefaultName>> contract_names, \
            std::unordered_set<DefaultName> fuse_names) { \
            return tensor_1.contract(tensor_2, std::move(contract_names), std::move(fuse_names)); \
         }, \
         py::arg("another_tensor"), \
         py::arg("contract_names"), \
         py::arg("fuse_names") = py::set(), \
         "Contract two tensors")
         tensor_t.TAT_LOOP_CONTRACT(float).TAT_LOOP_CONTRACT(double).TAT_LOOP_CONTRACT(std::complex<float>).TAT_LOOP_CONTRACT(std::complex<double>);
#undef TAT_LOOP_CONTRACT
         tensor_t
               .def(
                     "identity",
                     [](T& tensor, std::unordered_set<std::pair<DefaultName, DefaultName>>& pairs) -> T& {
                        return tensor.identity(pairs);
                     },
                     py::arg("pairs"),
                     "Get a identity tensor with same shape",
                     py::return_value_policy::reference_internal)
               .def("exponential", &T::exponential, py::arg("pairs"), py::arg("step") = 2, "Calculate exponential like matrix")
               .def("conjugate",
                    &T::conjugate,
                    py::arg("default_is_physics_edge") = false,
                    py::arg("exclude_names_set") = py::set(),
                    "Get the conjugate Tensor")
               .def("trace", &T::trace, py::arg("trace_names"), py::arg("fuse_names") = py::dict(), "Calculate trace or partial trace of a tensor")
               .def(
                     "svd",
                     [](const T& tensor,
                        const std::unordered_set<DefaultName>& free_name_set_u,
                        const DefaultName& common_name_u,
                        const DefaultName& common_name_v,
                        const DefaultName& singular_name_u,
                        const DefaultName& singular_name_v,
                        int cut,
                        double relative_cut,
                        double temperature) {
                        Cut real_cut = NoCut();
                        if (temperature > 0) { // T = 0 => RemainCut
                           real_cut = BoltzmannCut(temperature, cut, &random_engine);
                        } else if (relative_cut > 0) {
                           real_cut = RelativeCut(relative_cut);
                        } else if (cut > 0) {
                           real_cut = RemainCut(cut);
                        }
                        auto result = tensor.svd(free_name_set_u, common_name_u, common_name_v, singular_name_u, singular_name_v, real_cut);
                        return py::make_tuple(std::move(result.U), std::move(result.S), std::move(result.V));
                     },
                     py::arg("free_name_set_u"),
                     py::arg("common_name_u"),
                     py::arg("common_name_v"),
                     py::arg("singular_name_u"),
                     py::arg("singular_name_v"),
                     py::arg("cut") = 0,
                     py::arg("relative_cut") = 0,
                     py::arg("temperature") = 0,
                     "Singular value decomposition")
               .def(
                     "qr",
                     [](const T& tensor,
                        char free_name_direction,
                        const std::unordered_set<DefaultName>& free_name_set,
                        const DefaultName& common_name_q,
                        const DefaultName& common_name_r) {
                        auto result = tensor.qr(free_name_direction, free_name_set, common_name_q, common_name_r);
                        return py::make_tuple(std::move(result.Q), std::move(result.R));
                     },
                     py::arg("free_name_direction"),
                     py::arg("free_name_set"),
                     py::arg("common_name_q"),
                     py::arg("common_name_r"),
                     "QR decomposition")
               .def("shrink",
                    &T::shrink,
                    "Shrink Edge of tensor",
                    py::arg("configure"),
                    py::arg("new_name") = InternalName<DefaultName>::No_New_Name,
                    py::arg("arrow") = false)
               .def("expand", &T::expand, "Expand Edge of tensor", py::arg("configure"), py::arg("old_name") = InternalName<DefaultName>::No_Old_Name)
               .def(
                     "dump",
                     [](const T& tensor) {
                        return py::bytes(tensor.dump());
                     },
                     "dump Tensor to bytes")
               .def(
                     "load",
                     [](T& tensor, const py::bytes& bytes) -> T& {
                        return tensor.load(std::string(bytes));
                     },
                     "Load Tensor from bytes",
                     py::return_value_policy::reference_internal)
               .def(py::pickle(
                     [](const T& tensor) {
                        return py::bytes(tensor.dump());
                     },
                     [](const py::bytes& bytes) {
                        return T().load(std::string(bytes));
                     }))
               .def(
                     "rand",
                     [](T& tensor, ScalarType min, ScalarType max) -> T& {
                        if constexpr (is_complex<ScalarType>) {
                           auto distribution_real = std::uniform_real_distribution<real_scalar<ScalarType>>(min.real(), max.real());
                           auto distribution_imag = std::uniform_real_distribution<real_scalar<ScalarType>>(min.imag(), max.imag());
                           return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                              return {distribution_real(random_engine), distribution_imag(random_engine)};
                           });
                        } else {
                           auto distribution = std::uniform_real_distribution<real_scalar<ScalarType>>(min, max);
                           return tensor.set([&distribution]() {
                              return distribution(random_engine);
                           });
                        }
                     },
                     py::arg("min") = 0,
                     py::arg("max") = one,
                     "Set Uniform Random Number into Tensor",
                     py::return_value_policy::reference_internal)
               .def(
                     "randn",
                     [](T& tensor, ScalarType mean, ScalarType stddev) -> T& {
                        if constexpr (is_complex<ScalarType>) {
                           auto distribution_real = std::normal_distribution<real_scalar<ScalarType>>(mean.real(), stddev.real());
                           auto distribution_imag = std::normal_distribution<real_scalar<ScalarType>>(mean.imag(), stddev.imag());
                           return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                              return {distribution_real(random_engine), distribution_imag(random_engine)};
                           });
                        } else {
                           auto distribution = std::normal_distribution<real_scalar<ScalarType>>(mean, stddev);
                           return tensor.set([&distribution]() {
                              return distribution(random_engine);
                           });
                        }
                     },
                     py::arg("mean") = 0,
                     py::arg("stddev") = one,
                     "Set Normal Distribution Random Number into Tensor",
                     py::return_value_policy::reference_internal);
      };
   }

   template<typename Symmetry, bool IsSegmentNotSymmetries, typename Element, bool IsTuple, typename Input>
   auto convert_element_to_symmetry(const Input& input) {
      if constexpr (IsSegmentNotSymmetries) {
         auto segments = std::vector<std::pair<Symmetry, Size>>();
         for (const auto& [key, value] : input) {
            if constexpr (IsTuple) {
               segments.push_back({std::make_from_tuple<Symmetry>(key), value});
            } else {
               segments.push_back({Symmetry(key), value});
            }
         }
         return segments;
      } else {
         auto symmetries = std::vector<Symmetry>();
         for (const auto& element : input) {
            if constexpr (IsTuple) {
               symmetries.push_back(std::make_from_tuple<Symmetry>(element));
            } else {
               // element maybe bit reference from vector bool
               if constexpr (std::is_base_of_v<std::tuple<bool>, Symmetry>) {
                  symmetries.push_back(Symmetry(bool(element)));
               } else {
                  symmetries.push_back(Symmetry(element));
               }
            }
         }
         return symmetries;
      }
   }

   template<typename Symmetry, typename Element, bool IsTuple, template<typename, bool = false> class EdgeType = Edge>
   auto declare_edge(py::module_& symmetry_m, const char* name) {
      // EdgeType maybe edge_segment_t or Edge directly
      // Element is implicitly convertible to Symmetry
      // IsTuple describe wether Element is a tuple

      constexpr bool need_arrow = Symmetry::is_fermi_symmetry;
      constexpr bool need_element = !std::is_same_v<Element, void>;
      constexpr bool real_edge = is_edge<EdgeType<Symmetry>>;
      // need_arrow no, need element no
      // need_arrow no, need_element yes
      // need_arrow yes, need element yes

      auto result = py::class_<EdgeType<Symmetry>>(
                          symmetry_m,
                          real_edge ? "Edge" : "EdgeSegment",
                          ("Edge with symmetry type as " + std::string(name) + "Symmetry").c_str())
                          .def(implicit_init<EdgeType<Symmetry>, Size>(), py::arg("dimension"), "Edge with only one trivial segment")
                          .def_property_readonly(
                                "segment",
                                [](const EdgeType<Symmetry>& edge) {
                                   return edge.segments();
                                })
                          .def_property_readonly("dimension", &EdgeType<Symmetry>::total_dimension)
                          .def("conjugated", &EdgeType<Symmetry>::conjugated, "Get conjugated edge of this edge")
                          .def("get_point_from_index", &EdgeType<Symmetry>::point_by_index, "Get edge point from index")
                          .def("get_index_from_point", &EdgeType<Symmetry>::index_by_point, "Get index from edge point")
                          .def(py::self == py::self)
                          .def(py::self != py::self);

      if constexpr (real_edge) {
         if constexpr (need_arrow) {
            result.def_property_readonly("arrow", &EdgeType<Symmetry>::arrow, "Fermi Arrow of the edge");
         } else {
            result.def_property_readonly_static("arrow", &EdgeType<Symmetry>::arrow, "Boson Arrow of the edge, always False");
         }
      }

      if constexpr (real_edge) {
         result.def(py::pickle(
               [](const EdgeType<Symmetry>& edge) {
                  auto out = std::stringstream();
                  out < edge;
                  return py::bytes(out.str());
               },
               [](const py::bytes& bytes) {
                  EdgeType<Symmetry> edge;
                  auto in = std::stringstream(std::string(bytes));
                  in > edge;
                  return edge;
               }));
      }

      if constexpr (real_edge) {
         // __str__ and __repr__
         result.def("__str__",
                    [](const EdgeType<Symmetry>& edge) {
                       auto out = std::stringstream();
                       out << edge;
                       return out.str();
                    })
               .def("__repr__", [name](const EdgeType<Symmetry>& edge) {
                  auto out = std::stringstream();
                  out << name << "Edge";
                  if constexpr (Symmetry::length == 0) {
                     out << "[";
                  }
                  out << edge;
                  if constexpr (Symmetry::length == 0) {
                     out << "]";
                  }
                  return out.str();
               });
      }

      // non trivial constructor
      // trivial single segment has already defined, it is ususally used in NoSymmetry

      // [(Sym, Size)]
      // [Sym]

      // []
      // [], Arrow
      // ([], Arrow)

      // Sym
      // Ele

      // [(Sym, Size)] * []
      result.def(
            implicit_init<EdgeType<Symmetry>, std::vector<std::pair<Symmetry, Size>>>(),
            py::arg("segments"),
            "Create Edge with list of pair of symmetry and dimension");
      if constexpr (need_element) {
         result.def(
               implicit_init<EdgeType<Symmetry>, std::vector<std::pair<Element, Size>>>(
                     [](const std::vector<std::pair<Element, Size>>& element_segment) {
                        return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, true, Element, IsTuple>(element_segment));
                     }),
               py::arg("segments"),
               "Create Edge with list of pair of symmetry and dimension");
      }
      if constexpr (real_edge) {
         // [(Sym, Size)] * [], Arrow
         result.def(
               py::init<std::vector<std::pair<Symmetry, Size>>, Arrow>(),
               py::arg("segments"),
               py::arg("arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  py::init([](const std::vector<std::pair<Element, Size>>& element_segment, Arrow arrow) {
                     return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, true, Element, IsTuple>(element_segment), arrow);
                  }),
                  py::arg("segments"),
                  py::arg("arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
         // [(Sym, Size)] * ([], Arrow)
         result.def(
               implicit_init<EdgeType<Symmetry>, std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow>>(
                     [](std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow> p) {
                        return std::make_from_tuple<EdgeType<Symmetry>>(std::move(p));
                     }),
               py::arg("pair_of_segments_and_arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  implicit_init<EdgeType<Symmetry>, std::pair<std::vector<std::pair<Element, Size>>, Arrow>>(
                        [](const std::pair<std::vector<std::pair<Element, Size>>, Arrow>& p) {
                           const auto& [element_segment, arrow] = p;
                           return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, true, Element, IsTuple>(element_segment), arrow);
                        }),
                  py::arg("pair_of_segments_and_arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
      }
      // [Sym] * []
      result.def(
            implicit_init<EdgeType<Symmetry>, const std::vector<Symmetry>&>(),
            py::arg("symmetries"),
            "Create Edge with list of symmetries which construct several one dimension segments");
      if constexpr (need_element) {
         result.def(
               implicit_init<EdgeType<Symmetry>, const std::vector<Element>&>([](const std::vector<Element>& element_symmetries) {
                  return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, false, Element, IsTuple>(element_symmetries));
               }),
               py::arg("symmetries"),
               "Create Edge with list of symmetries which construct several one dimension segments");
      }
      if constexpr (real_edge) {
         // [Sym] * [], Arrow
         result.def(
               py::init<std::vector<Symmetry>, Arrow>(),
               py::arg("symmetries"),
               py::arg("arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  py::init([](const std::vector<Element>& element_symmetries, Arrow arrow) {
                     return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, false, Element, IsTuple>(element_symmetries), arrow);
                  }),
                  py::arg("symmetries"),
                  py::arg("arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
         // [Sym] * ([], Arrow)
         result.def(
               implicit_init<EdgeType<Symmetry>, std::pair<std::vector<Symmetry>, Arrow>>(
                     [](std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow> p) {
                        return std::make_from_tuple<EdgeType<Symmetry>>(std::move(p));
                     }),
               py::arg("pair_of_symmetries_and_arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  implicit_init<EdgeType<Symmetry>, std::pair<std::vector<Element>, Arrow>>([](const std::pair<std::vector<Element>, Arrow>& p) {
                     const auto& [element_symmetries, arrow] = p;
                     return EdgeType<Symmetry>(convert_element_to_symmetry<Symmetry, false, Element, IsTuple>(element_symmetries), arrow);
                  }),
                  py::arg("pair_of_symmetries_and_arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
      }

      return result;
   }

   template<typename Symmetry>
   auto declare_symmetry(py::module_& symmetry_m, const char* name) {
      // symmetry_m: TAT.Z2/TAT.No/TAT.Fermi/...
      // define TAT.Fermi.Symmetry as FermiSymmetry in this function
      // it does not define constructor, it is needed to define constructor later
      return py::class_<Symmetry>(symmetry_m, "Symmetry", (std::string(name) + "Symmetry").c_str())
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(-py::self)
            .def("__hash__",
                 [](const Symmetry& symmetry) {
                    return py::hash(py::cast(static_cast<const typename Symmetry::base_tuple_t&>(symmetry)));
                 })
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
            .def("__str__",
                 [=](const Symmetry& symmetry) {
                    auto out = std::stringstream();
                    out << symmetry;
                    return out.str();
                 })
            .def(py::pickle(
                  [](const Symmetry& symmetry) {
                     auto out = std::stringstream();
                     out < symmetry;
                     return py::bytes(out.str());
                  },
                  [](const py::bytes& bytes) {
                     Symmetry symmetry;
                     auto in = std::stringstream(std::string(bytes));
                     in > symmetry;
                     return symmetry;
                  }));
   }
} // namespace TAT

#endif
