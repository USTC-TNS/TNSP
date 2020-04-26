#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define TAT_ALWAYS_COLOR
#include "TAT/TAT.hpp"

namespace TAT {
   namespace py = pybind11;

   template<class Type, class Args>
   auto implicit_init() {
      py::implicitly_convertible<Args, Type>();
      return py::init<Args>();
   }

   template<class Type, class Args, class Func>
   auto implicit_init(Func&& func) {
      py::implicitly_convertible<Args, Type>();
      return py::init(func);
   }

   template<class ScalarType, class Symmetry>
   auto declare_tensor(py::module& m, const char* name) {
      using T = Tensor<ScalarType, Symmetry>;
      using E = Edge<Symmetry>;
      return py::class_<T>(m, name)
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
            .def("shape",
                 [](const T& tensor) {
                    auto out = std::stringstream();
                    out << '{' << console_green << "names" << console_origin << ':';
                    out << tensor.names;
                    out << ',' << console_green << "edges" << console_origin << ':';
                    out << tensor.core->edges;
                    out << '}';
                    return out.str();
                 })
            .def(py::init<const std::vector<Name>&, const std::vector<E>&>())
            .def(implicit_init<T, std::tuple<std::vector<Name>, std::vector<E>>>(
                  [](const std::tuple<std::vector<Name>, std::vector<E>>& p) { return std::make_from_tuple<T>(p); }))
            // problem: `TAT.TensorDoubleU1Symmetry(list("AB"),[{0:2},{0:2}])` not work
            // https://github.com/pybind/pybind11/issues/2182
            // 无法迭代的隐式初始化, boost::python可以但是需要对stl的各个容器自己做适配，太难了
            // 用 pybind11 的话可以做的是TEdge直接由U1/Z2/Fermi/...构建的constructor
            .def(implicit_init<T, ScalarType>())
            .def("value", [](const T& tensor) -> ScalarType { return tensor; })
            .def("copy", &T::copy)
            .def("same_shape", &T::same_shape)
            .def("map", [](const T& tensor, std::function<ScalarType(ScalarType)>& function) { return tensor.map(function); })
            .def("transform", [](T& tensor, std::function<ScalarType(ScalarType)>& function) -> T& { return tensor.transform(function); })
            .def("set", [](T& tensor, std::function<ScalarType()>& function) -> T& { return tensor.set(function); })
            .def("zero", [](T& tensor) -> T& { return tensor.zero(); })
            .def(
                  "test",
                  [](T& tensor, ScalarType first, ScalarType step) -> T& { return tensor.test(first, step); },
                  py::arg("first") = 0,
                  py::arg("step") = 1)
            .def("block", [](T& tensor, const std::map<Name, Symmetry>& position) { return tensor.block(position); })
            .def("at", [](T& tensor, const std::map<Name, typename T::EdgeInfoForGetItem>& position) { return tensor.at(position); })
            .def("toS", &T::template to<float>)
            .def("toD", &T::template to<double>)
            .def("toC", &T::template to<std::complex<float>>)
            .def("toZ", &T::template to<std::complex<double>>)
            .def("norm_max", &T::template norm<-1>)
            .def("norm_num", &T::template norm<0>)
            .def("norm_1", &T::template norm<1>)
            .def("norm_2", &T::template norm<2>)
            .def("edge_rename", &T::edge_rename)
            .def("transpose", [](const T& tensor, const std::vector<Name>& names) { return tensor.transpose(names); })
            .def("reverse_edge", &T::reverse_edge)
            .def("merge_edge", &T::merge_edge)
            .def("split_edge", &T::split_edge)
            .def("contract",
                 [](const T& tensor_1, const T& tensor_2, std::set<std::tuple<Name, Name>> contract_names) {
                    return T::contract(tensor_1, tensor_2, contract_names);
                 })
            .def("contract_all_edge", [](const T& tensor) { return tensor.contract_all_edge(); })
            .def("contract_all_edge", [](const T& tensor, const T& other) { return tensor.contract_all_edge(other); })
            .def("trace", &T::trace)
            .def("conjugate", &T::conjugate)
            // multiple svd slice
            ;
   }

   template<class Symmetry, class Element, bool IsTuple>
   auto declare_edge(py::module& m, const char* name) {
      auto result = py::class_<Edge<Symmetry>>(m, name)
                          .def_readwrite("map", &Edge<Symmetry>::map)
                          .def(implicit_init<Edge<Symmetry>, Size>())
                          .def(implicit_init<Edge<Symmetry>, std::map<Symmetry, Size>>())
                          .def(implicit_init<Edge<Symmetry>, const std::set<Symmetry>&>())
                          .def("__str__",
                               [](const Edge<Symmetry>& edge) {
                                  auto out = std::stringstream();
                                  out << edge;
                                  return out.str();
                               })
                          .def("__repr__", [](const Edge<Symmetry>& edge) {
                             auto out = std::stringstream();
                             out << "Edge";
                             out << edge;
                             return out.str();
                          });
      if constexpr (!std::is_same_v<Element, void>) {
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
                        }))
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
                        }));
      }
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         result = result.def_readwrite("arrow", &Edge<Symmetry>::arrow)
                        .def(py::init<Arrow, std::map<Symmetry, Size>>())
                        .def(implicit_init<Edge<Symmetry>, std::tuple<Arrow, std::map<Symmetry, Size>>>(
                              [](const std::tuple<Arrow, std::map<Symmetry, Size>>& p) { return std::make_from_tuple<Edge<Symmetry>>(p); }));
         if constexpr (!std::is_same_v<Element, void>) {
            result = result.def(
                  implicit_init<Edge<Symmetry>, std::tuple<Arrow, std::map<Element, Size>>>([](const std::tuple<Arrow, std::map<Element, Size>>& p) {
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
                  }));
         }
      }
      return result;
   }

   template<class Symmetry>
   auto declare_symmetry(py::module& m, const char* name) {
      return py::class_<Symmetry>(m, name)
            .def("__repr__",
                 [=](const Symmetry& symmetry) {
                    auto out = std::stringstream();
                    out << name;
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
      // name
      py::class_<Name>(m, "Name")
            .def(implicit_init<Name, const std::string&>())
            .def("__repr__", [](const Name& name) { return "Name[" + id_to_name.at(name.id) + "]"; })
            .def("__str__", [](const Name& name) { return id_to_name.at(name.id); })
            .def_readonly("id", &Name::id)
            .def_property_readonly("name", [](const Name& name) { return id_to_name.at(name.id); });

      // symmetry
      m.attr("Symmetry") = py::dict(
            py::arg("NoSymmetry") = declare_symmetry<NoSymmetry>(m, "NoSymmetry").def(py::init<>()),
            py::arg("Z2Symmetry") =
                  declare_symmetry<Z2Symmetry>(m, "Z2Symmetry").def(implicit_init<Z2Symmetry, Z2>()).def_readwrite("z2", &Z2Symmetry::z2),
            py::arg("U1Symmetry") =
                  declare_symmetry<U1Symmetry>(m, "U1Symmetry").def(implicit_init<U1Symmetry, U1>()).def_readwrite("u1", &U1Symmetry::u1),
            py::arg("FermiSymmmetry") = declare_symmetry<FermiSymmetry>(m, "FermiSymmetry")
                                              .def(implicit_init<FermiSymmetry, Fermi>())
                                              .def_readwrite("fermi", &FermiSymmetry::fermi),
            py::arg("FermiZ2Symmmetry") = declare_symmetry<FermiZ2Symmetry>(m, "FermiZ2Symmetry")
                                                .def(py::init<Fermi, Z2>())
                                                .def(implicit_init<FermiZ2Symmetry, const std::tuple<Fermi, Z2>&>(
                                                      [](const std::tuple<Fermi, Z2>& p) { return std::make_from_tuple<FermiZ2Symmetry>(p); }))
                                                .def_readwrite("fermi", &FermiZ2Symmetry::fermi)
                                                .def_readwrite("z2", &FermiZ2Symmetry::z2),
            py::arg("FermiU1Symmmetry") = declare_symmetry<FermiU1Symmetry>(m, "FermiU1Symmetry")
                                                .def(py::init<Fermi, U1>())
                                                .def(implicit_init<FermiU1Symmetry, const std::tuple<Fermi, U1>&>(
                                                      [](const std::tuple<Fermi, U1>& p) { return std::make_from_tuple<FermiU1Symmetry>(p); }))
                                                .def_readwrite("fermi", &FermiU1Symmetry::fermi)
                                                .def_readwrite("u1", &FermiU1Symmetry::u1));
      // edge
      m.attr("Edge") = py::dict(
            py::arg("NoSymmetry") = declare_edge<NoSymmetry, void, false>(m, "EdgeNoSymmetry"),
            py::arg("Z2Symmetry") = declare_edge<Z2Symmetry, Z2, false>(m, "EdgeZ2Symmetry"),
            py::arg("U1Symmetry") = declare_edge<U1Symmetry, U1, false>(m, "EdgeU1Symmetry"),
            py::arg("FermiSymmetry") = declare_edge<FermiSymmetry, Fermi, false>(m, "EdgeFermiSymmetry"),
            py::arg("FermiZ2Symmetry") = declare_edge<FermiZ2Symmetry, std::tuple<Fermi, Z2>, true>(m, "EdgeFermiZ2Symmetry"),
            py::arg("FermiU1Symmetry") = declare_edge<FermiU1Symmetry, std::tuple<Fermi, U1>, true>(m, "EdgeFermiU1Symmetry"));
      // tensor
#define DECLARE_TENSOR(SCALAR, SCALARNAME, SYMMETRY) py::arg(#SYMMETRY) = declare_tensor<SCALAR, SYMMETRY>(m, "Tensor" SCALARNAME #SYMMETRY)
#define DECLARE_TENSOR_WITH_SAME_SCALAR(SCALAR, SCALARNAME)   \
   py::dict(                                                  \
         DECLARE_TENSOR(SCALAR, SCALARNAME, NoSymmetry),      \
         DECLARE_TENSOR(SCALAR, SCALARNAME, Z2Symmetry),      \
         DECLARE_TENSOR(SCALAR, SCALARNAME, U1Symmetry),      \
         DECLARE_TENSOR(SCALAR, SCALARNAME, FermiSymmetry),   \
         DECLARE_TENSOR(SCALAR, SCALARNAME, FermiZ2Symmetry), \
         DECLARE_TENSOR(SCALAR, SCALARNAME, FermiU1Symmetry))
#define DECLARE_TENSOR_DICT(SCALAR, SCALARNAME) py::arg(#SCALAR) = DECLARE_TENSOR_WITH_SAME_SCALAR(SCALAR, SCALARNAME)
      m.attr("Tensor") = py::dict(
            DECLARE_TENSOR_DICT(float, "S"),
            DECLARE_TENSOR_DICT(double, "D"),
            DECLARE_TENSOR_DICT(std::complex<float>, "C"),
            DECLARE_TENSOR_DICT(std::complex<double>, "Z"));
#undef DECLARE_TENSOR_WITH_SAME_SCALAR
#undef DECLARE_TENSOR
   }
} // namespace TAT
