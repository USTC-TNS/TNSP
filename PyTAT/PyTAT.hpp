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

#ifndef PYTAT_HPP
#define PYTAT_HPP

#include <functional>
#include <random>
#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
   // Auxiliaries
   struct AtExit {
      std::vector<std::function<void()>> function_list;
      void operator()(std::function<void()>&& function) {
         function_list.push_back(std::move(function));
      }
      void release() {
         for (auto i = 0; i < function_list.size(); i++) {
            // Use index to avoid crash if function_list is modified when calling function_list[i]
            function_list[i]();
         }
         function_list.clear();
      }
   };
   inline AtExit at_exit;

   template<typename Type, typename Input>
   auto implicit_init() {
      at_exit([]() {
         py::implicitly_convertible<Input, Type>();
      });
      return py::init<Input>();
   }

   template<typename Type, typename Input, typename Func>
   auto implicit_init(Func&& func) {
      at_exit([]() {
         py::implicitly_convertible<Input, Type>();
      });
      return py::init(func);
   }

   // About callable module
   inline void set_callable(py::module_& tat_m) {
      auto py_type = py::module_::import("builtins").attr("type");
      py::dict callable_type_dict;
      callable_type_dict["__call__"] = std::function([tat_m]() {
         return tat_m.attr("information");
      });
      py::list base_types;
      base_types.append(py::type::of(tat_m));
      tat_m.attr("__class__") = py_type("CallableModuleForTAT", py::tuple(base_types), callable_type_dict);
   }

   // About random
   inline auto random_engine = std::default_random_engine(std::random_device()());
   inline void set_random(py::module_& tat_m) {
      auto random_m = tat_m.def_submodule("random", "random for TAT");
      random_m.def(
            "seed",
            [](unsigned int seed) {
               random_engine.seed(seed);
            },
            "Set Random Seed",
            py::arg("seed"));
      random_m.def(
            "uniform_int",
            [](int min, int max) {
               return std::function([distribution = std::uniform_int_distribution<int>(min, max)]() mutable {
                  return distribution(random_engine);
               });
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform integer");
      random_m.def(
            "uniform_real",
            [](double min, double max) {
               return std::function([distribution = std::uniform_real_distribution<double>(min, max)]() mutable {
                  return distribution(random_engine);
               });
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform real");
      random_m.def(
            "normal",
            [](double mean, double stddev) {
               return std::function([distribution = std::normal_distribution<double>(mean, stddev)]() mutable {
                  return distribution(random_engine);
               });
            },
            py::arg("mean") = 0,
            py::arg("stddev") = 1,
            "Get random normal real");
   }

   // About symmetry
   template<typename Symmetry>
   auto dealing_symmetry(py::module_& symmetry_m, const char* name) {
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
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(-py::self)
            .def_property_readonly("parity", &Symmetry::parity)
            .def("__hash__",
                 [](const Symmetry& symmetry) {
                    // Use the hash for the hash of the base tuple for the symmetry type
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

   // About edge
   template<typename>
   struct is_tuple : std::false_type {};
   template<typename... Args>
   struct is_tuple<std::tuple<Args...>> : std::true_type {};
   template<typename T>
   constexpr bool is_tuple_v = is_tuple<T>::value;

   template<typename Symmetry, typename Input>
   auto convert_to_symmetry(Input input) {
      if constexpr (is_tuple_v<remove_cvref_t<Input>>) {
         return std::make_from_tuple<Symmetry>(input);
      } else if constexpr (std::is_base_of_v<std::tuple<bool>, Symmetry>) {
         // input may be bit reference in vector bool, cannot be converted to symmetry directly
         return Symmetry(bool(input));
      } else {
         return Symmetry(input);
      }
   }

   template<typename Symmetry, bool IsSegmentNotSymmetries, typename Input>
   auto convert_to_symmetries(const Input& input) {
      if constexpr (IsSegmentNotSymmetries) {
         auto segments = std::vector<std::pair<Symmetry, Size>>();
         for (const auto& [key, value] : input) {
            segments.push_back({convert_to_symmetry<Symmetry>(key), value});
         }
         return segments;
      } else {
         auto symmetries = std::vector<Symmetry>();
         for (const auto& element : input) {
            symmetries.push_back(convert_to_symmetry<Symmetry>(element));
         }
         return symmetries;
      }
   }

   template<typename T>
   struct element_of_symmetry {
      using type = T;
   };
   template<typename First>
   struct element_of_symmetry<std::tuple<First>> {
      using type = First;
   };
   template<>
   struct element_of_symmetry<std::tuple<>> {
      using type = void;
   };
   template<typename T>
   using element_of_symmetry_t = typename element_of_symmetry<T>::type;

   template<typename Symmetry, bool real_edge>
   auto dealing_edge(py::module_& symmetry_m, const char* name) {
      using BaseTuple = typename Symmetry::base_tuple_t;
      using Element = element_of_symmetry_t<BaseTuple>;
      // Element is implicitly convertible to Symmetry, except it is void for NoSymmetry

      using E = std::conditional_t<real_edge, Edge<Symmetry>, edge_segments_t<Symmetry>>;

      constexpr bool need_arrow = Symmetry::is_fermi_symmetry;
      constexpr bool need_element = !std::is_same_v<Element, void>;
      // need_arrow no, need element no
      // need_arrow no, need_element yes
      // need_arrow yes, need element yes

      auto result =
            py::class_<E>(symmetry_m, real_edge ? "Edge" : "EdgeSegment", ("Edge with symmetry type as " + std::string(name) + "Symmetry").c_str())
                  .def(implicit_init<E, Size>(), py::arg("dimension"), "Edge with only one trivial segment")
                  .def_property_readonly("segments", static_cast<const typename E::segments_t& (E::*)() const>(&E::segments))
                  .def_property_readonly("segments_size", &E::segments_size)
                  .def("coord_by_point", &E::coord_by_point)
                  .def("point_by_coord", &E::point_by_coord)
                  .def("coord_by_index", &E::coord_by_index)
                  .def("index_by_coord", &E::index_by_coord)
                  .def("point_by_index", &E::point_by_index)
                  .def("index_by_point", &E::index_by_point)
                  .def("position_by_symmetry", &E::position_by_symmetry)
                  .def("dimension_by_symmetry", &E::dimension_by_symmetry)
                  .def("conjugated", &E::conjugated, "Get conjugated edge of this edge")
                  .def_property_readonly("dimension", &E::total_dimension)
                  .def(py::self == py::self)
                  .def(py::self != py::self);

      // Real edge specific
      if constexpr (real_edge) {
         if constexpr (need_arrow) {
            result.def_property_readonly("arrow", &E::arrow, "Fermi Arrow of the edge");
         } else {
            result.attr("arrow") = E::arrow();
         }

         result.def(py::pickle(
               [](const E& edge) {
                  auto out = std::stringstream();
                  out < edge;
                  return py::bytes(out.str());
               },
               [](const py::bytes& bytes) {
                  E edge;
                  auto in = std::stringstream(std::string(bytes));
                  in > edge;
                  return edge;
               }));

         // __str__ and __repr__
         result.def("__str__",
                    [](const E& edge) {
                       auto out = std::stringstream();
                       out << edge;
                       return out.str();
                    })
               .def("__repr__", [name](const E& edge) {
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

      // Segments:
      // [(Sym, Size)]
      // [Sym]

      // Edge: [] means segments
      // []
      // [], Arrow   # only for real edge
      // ([], Arrow) # only for real edge

      // There are 3*2 = 6 type of non trivial constructor

      // Every type also need constructor directly from element instead of symmetry

      // [(Sym, Size)] * []
      result.def(
            implicit_init<E, std::vector<std::pair<Symmetry, Size>>>(),
            py::arg("segments"),
            "Create Edge with list of pair of symmetry and dimension");
      if constexpr (need_element) {
         result.def(
               implicit_init<E, std::vector<std::pair<Element, Size>>>([](const std::vector<std::pair<Element, Size>>& element_segments) {
                  return E(convert_to_symmetries<Symmetry, true>(element_segments));
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
                  py::init([](const std::vector<std::pair<Element, Size>>& element_segments, Arrow arrow) {
                     return E(convert_to_symmetries<Symmetry, true>(element_segments), arrow);
                  }),
                  py::arg("segments"),
                  py::arg("arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
         // [(Sym, Size)] * ([], Arrow)
         result.def(
               implicit_init<E, std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow>>(
                     [](std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow> p) {
                        return std::make_from_tuple<E>(std::move(p));
                     }),
               py::arg("pair_of_segments_and_arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  implicit_init<E, std::pair<std::vector<std::pair<Element, Size>>, Arrow>>(
                        [](const std::pair<std::vector<std::pair<Element, Size>>, Arrow>& p) {
                           const auto& [element_segments, arrow] = p;
                           return E(convert_to_symmetries<Symmetry, true>(element_segments), arrow);
                        }),
                  py::arg("pair_of_segments_and_arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
      }
      // [Sym] * []
      result.def(
            implicit_init<E, std::vector<Symmetry>>(),
            py::arg("symmetries"),
            "Create Edge with list of symmetries which construct several one dimension segments");
      if constexpr (need_element) {
         result.def(
               implicit_init<E, std::vector<Element>>([](const std::vector<Element>& element_symmetries) {
                  return E(convert_to_symmetries<Symmetry, false>(element_symmetries));
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
                     return E(convert_to_symmetries<Symmetry, false>(element_symmetries), arrow);
                  }),
                  py::arg("symmetries"),
                  py::arg("arrow"),
                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
         // [Sym] * ([], Arrow)
         result.def(
               implicit_init<E, std::pair<std::vector<Symmetry>, Arrow>>([](std::pair<std::vector<std::pair<Symmetry, Size>>, Arrow> p) {
                  return std::make_from_tuple<E>(std::move(p));
               }),
               py::arg("pair_of_symmetries_and_arrow"),
               "Edge created from segments and arrow, for boson edge, arrow will not be used");
         if constexpr (need_element) {
            result.def(
                  implicit_init<E, std::pair<std::vector<Element>, Arrow>>([](const std::pair<std::vector<Element>, Arrow>& p) {
                     const auto& [element_symmetries, arrow] = p;
                     return E(convert_to_symmetries<Symmetry, false>(element_symmetries), arrow);
                  }),

                  "Edge created from segments and arrow, for boson edge, arrow will not be used");
         }
      }

      return result;
   }

   // About tensor
   template<typename T>
   auto py_storage(T& tensor, py::object& base) {
      using ScalarType = typename T::scalar_t;
      auto& s = tensor.storage();
      return py::array_t<ScalarType>(
            py::buffer_info{
                  s.data(),
                  sizeof(ScalarType),
                  py::format_descriptor<ScalarType>::format(),
                  1,
                  std::vector<Size>{s.size()},
                  std::vector<Size>{sizeof(ScalarType)}},
            base);
   }

   template<typename T>
   auto py_mdspan(mdspan<T>& mdspan, const std::vector<DefaultName>& tensor_names, const std::vector<DefaultName>& names, py::object& base) {
      using ScalarType = T;
      std::vector<Size> dimensions;
      std::vector<Size> strides;
      dimensions.reserve(mdspan.rank());
      strides.reserve(mdspan.rank());
      for (auto i = 0; i < mdspan.rank(); i++) {
         auto j = std::distance(tensor_names.begin(), std::find(tensor_names.begin(), tensor_names.end(), names[i]));
         dimensions.push_back(mdspan.dimensions(j));
         strides.push_back(mdspan.leadings(j) * sizeof(ScalarType));
      }
      return py::array_t<ScalarType>(
            py::buffer_info{
                  mdspan.data(),
                  sizeof(ScalarType),
                  py::format_descriptor<ScalarType>::format(),
                  mdspan.rank(),
                  std::move(dimensions),
                  std::move(strides)},
            base);
   }

   template<typename ScalarType, typename Symmetry>
   struct blocks_of_tensor {
      py::object tensor;

      blocks_of_tensor(py::object& t) : tensor(t) {}

      auto blocks(const std::vector<std::pair<DefaultName, Symmetry>>& position) {
         auto& t = py::cast<Tensor<ScalarType, Symmetry, DefaultName>&>(tensor);
         std::unordered_map<DefaultName, Symmetry> map;
         std::vector<DefaultName> names;
         for (const auto& [name, symmetry] : position) {
            map[name] = symmetry;
            names.push_back(name);
         }

         return py_mdspan(t.blocks(map), t.names(), names, tensor);
      }

      auto blocks(const std::vector<DefaultName>& position) {
         auto& t = py::cast<Tensor<ScalarType, Symmetry, DefaultName>&>(tensor);
         std::unordered_map<DefaultName, Symmetry> map;
         for (const auto& name : position) {
            map[name] = Symmetry();
         }
         const auto& names = position;

         return py_mdspan(t.blocks(map), t.names(), names, tensor);
      }
   };

   template<typename ScalarType, typename Symmetry>
   auto dealing_tensor(
         py::module_& symmetry_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      auto self_m = symmetry_m.def_submodule(scalar_short_name.c_str());
      using T = Tensor<ScalarType, Symmetry>;
      using B = blocks_of_tensor<ScalarType, Symmetry>;
      using E = Edge<Symmetry>;
      std::string tensor_name = scalar_short_name + symmetry_short_name;

      py::class_<B>(
            self_m,
            "_Blocks",
            ("Blocks of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def("__getitem__",
                 [](B& b, const std::vector<std::pair<DefaultName, Symmetry>>& position) {
                    return b.blocks(position);
                 })
            .def("__getitem__",
                 [](B& b, const std::vector<DefaultName>& position) {
                    return b.blocks(position);
                 })
            .def("__setitem__",
                 [](B& b, const std::vector<std::pair<DefaultName, Symmetry>>& position, py::object& value) {
                    b.blocks(position).attr("__setitem__")(py::ellipsis(), value);
                 })
            .def("__setitem__", [](B& b, const std::vector<DefaultName>& position, py::object& value) {
               b.blocks(position).attr("__setitem__")(py::ellipsis(), value);
            });

      // one is used in default random distribution
      constexpr ScalarType one = []() constexpr {
         if constexpr (is_complex<ScalarType>) {
            return ScalarType(1, 1);
         } else {
            return 1;
         }
      }();
      auto tensor_t = py::class_<T>(
            self_m,
            "Tensor",
            ("Tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str());
      tensor_t.attr("model") = symmetry_m;

      // Define tensor function after all tensor has been declared.
      return [=]() mutable {
         tensor_t
               .def_property_readonly(
                     "names",
                     [](const T& tensor) {
                        return tensor.names();
                     })
               .def_property_readonly(
                     "rank",
                     [](const T& tensor) {
                        return tensor.rank();
                     })
               .def(
                     "edges",
                     [](const T& tensor, Rank r) -> const E& {
                        return tensor.edges(r);
                     },
                     py::return_value_policy::reference_internal)
               .def(
                     "edges",
                     [](const T& tensor, DefaultName r) -> const E& {
                        return tensor.edges(r);
                     },
                     py::return_value_policy::reference_internal)
               .def_property(
                     "storage",
                     [](py::object& tensor) {
                        auto& t = py::cast<Tensor<ScalarType, Symmetry, DefaultName>&>(tensor);
                        return py_storage(t, tensor);
                     },
                     [](py::object& tensor, py::object& value) {
                        auto& t = py::cast<Tensor<ScalarType, Symmetry, DefaultName>&>(tensor);
                        py_storage(t, tensor).attr("__setitem__")(py::ellipsis(), value);
                     })
               .def_property_readonly(
                     "blocks",
                     [](py::object& tensor) {
                        return blocks_of_tensor<ScalarType, Symmetry>(tensor);
                     })
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
                     py::arg("value"));

         tensor_t.def(ScalarType() + py::self)
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
         }
#undef TAT_LOOP_OPERATOR
         if constexpr (is_complex<ScalarType>) {
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
         tensor_t.def_readonly_static("is_real", &is_real<ScalarType>).def_readonly_static("is_complex", &is_complex<ScalarType>);

         tensor_t
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
                     }));

         tensor_t.def(py::init<>(), "Default Constructor")
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
                    "Read tensor from text string");

         tensor_t.def("copy", &T::copy, "Deep copy a tensor")
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
                     "reciprocal",
                     [](const T& tensor) {
                        return tensor.map([](ScalarType value) {
                           constexpr ScalarType zero = 0;
                           constexpr ScalarType one = 1;
                           return value == zero ? zero : one / value;
                        });
                     },
                     "Get elementwise reciprocal except zero")
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
                     "to",
                     [](const T& tensor, const py::object& object) -> py::object {
                        auto string = py::str(object);
                        auto contain = [&string](const char* other) {
                           return py::cast<bool>(string.attr("__contains__")(other));
                        };
                        if (contain("float32")) {
                           return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                        }
                        if (contain("float64")) {
                           return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                        }
                        if (contain("float")) {
                           return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                        }
                        if (contain("complex64")) {
                           return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                        }
                        if (contain("complex128")) {
                           return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                        }
                        if (contain("complex")) {
                           return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                        }
                        if (contain("S")) {
                           return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                        }
                        if (contain("D")) {
                           return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                        }
                        if (contain("C")) {
                           return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                        }
                        if (contain("Z")) {
                           return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                        }
                        throw std::runtime_error("Invalid scalar type in type conversion");
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
                     py::arg("dictionary"),
                     "Rename names of edges, which will not copy data")
               .def("transpose", &T::transpose, py::arg("target_names"), "Transpose the tensor to the order of new names")
               .def("reverse_edge",
                    &T::reverse_edge,
                    py::arg("reversed_names"),
                    py::arg("apply_parity") = false,
                    py::arg("parity_exclude_names") = py::set(),
                    "Reverse fermi arrow of several edge")
               .def("merge_edge",
                    &T::merge_edge,
                    py::arg("merge"),
                    py::arg("apply_parity") = false,
                    py::arg("parity_exclude_names_merge") = py::set(),
                    py::arg("parity_exclude_names_reverse") = py::set(),
                    "Merge several edges of the tensor into ones")
               .def("split_edge",
                    &T::split_edge,
                    py::arg("split"),
                    py::arg("apply_parity") = false,
                    py::arg("parity_exclude_names_split") = py::set(),
                    "Split edges of a tensor to many edges")
               .def("edge_operator",
                    &T::edge_operator,
                    py::arg("split_map"),
                    py::arg("reversed_names"),
                    py::arg("merge_map"),
                    py::arg("new_names"),
                    py::arg("apply_parity") = false,
                    py::arg("parity_exclude_names_split") = py::set(),
                    py::arg("parity_exclude_names_reverse_before_transpose") = py::set(),
                    py::arg("parity_exclude_names_reverse_after_transpose") = py::set(),
                    py::arg("parity_exclude_names_merge") = py::set(),
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
         py::arg("contract_pairs"), \
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
                    py::arg("exclude_names") = py::set(),
                    "Get the conjugate Tensor")
               .def("trace", &T::trace, py::arg("trace_pairs"), py::arg("fuse_names") = py::dict(), "Calculate trace or partial trace of a tensor")
               .def(
                     "svd",
                     [](const T& tensor,
                        const std::unordered_set<DefaultName>& free_names_u,
                        const DefaultName& common_name_u,
                        const DefaultName& common_name_v,
                        const DefaultName& singular_name_u,
                        const DefaultName& singular_name_v,
                        double cut) {
                        Cut real_cut = NoCut();
                        if (cut > 0) {
                           if (cut >= 1) {
                              real_cut = RemainCut(Size(cut));
                           } else {
                              real_cut = RelativeCut(cut);
                           }
                        }
                        auto result = tensor.svd(free_names_u, common_name_u, common_name_v, singular_name_u, singular_name_v, real_cut);
                        return py::make_tuple(std::move(result.U), std::move(result.S), std::move(result.V));
                     },
                     py::arg("free_names_u"),
                     py::arg("common_name_u"),
                     py::arg("common_name_v"),
                     py::arg("singular_name_u"),
                     py::arg("singular_name_v"),
                     py::arg("cut") = -1,
                     "Singular value decomposition")
               .def(
                     "qr",
                     [](const T& tensor,
                        char free_names_direction,
                        const std::unordered_set<DefaultName>& free_names,
                        const DefaultName& common_name_q,
                        const DefaultName& common_name_r) {
                        auto result = tensor.qr(free_names_direction, free_names, common_name_q, common_name_r);
                        return py::make_tuple(std::move(result.Q), std::move(result.R));
                     },
                     py::arg("free_names_direction"),
                     py::arg("free_names"),
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
} // namespace TAT

#endif
