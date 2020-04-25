#include <sstream>

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
            .def(implicit_init<T, std::pair<std::vector<Name>, std::vector<E>>>([](const std::pair<std::vector<Name>, std::vector<E>>& p) {
               return std::make_unique<T>(p.first, p.second);
            }))
            // TODO problem: `TAT.TensorDoubleU1Symmetry(list("AB"),[{0:2},{0:2}])` not work
            // https://github.com/pybind/pybind11/issues/2182
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
            // TODO block and at
            // TODO to
            .def("norm_max", &T::template norm<-1>)
            .def("norm_num", &T::template norm<0>)
            .def("norm_1", &T::template norm<1>)
            .def("norm_2", &T::template norm<2>);
      // TODO edge contract svd
   }

   template<class Symmetry>
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
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         result = result.def_readwrite("arrow", &Edge<Symmetry>::arrow).def(py::init<Arrow, std::map<Symmetry, Size>>());
      }
      return result;
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
            py::arg("NoSymmetry") = py::class_<NoSymmetry>(m, "NoSymmetry")
                                          .def(py::init<>())
                                          .def("__repr__", [=](const NoSymmetry& symmetry) { return std::string("NoSymmetry") + "[" + +"]"; })
                                          .def("__str__",
                                               [=](const NoSymmetry& symmetry) {
                                                  auto out = std::stringstream();
                                                  out << symmetry;
                                                  return out.str();
                                               }),
            py::arg("Z2Symmetry") =
                  py::class_<Z2Symmetry>(m, "Z2Symmetry")
                        .def(implicit_init<Z2Symmetry, Z2>())
                        .def("__repr__",
                             [=](const Z2Symmetry& symmetry) { return std::string("Z2Symmetry") + "[" + std::to_string(symmetry.z2) + "]"; })
                        .def("__str__",
                             [=](const Z2Symmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("z2", &Z2Symmetry::z2),
            py::arg("U1Symmetry") =
                  py::class_<U1Symmetry>(m, "U1Symmetry")
                        .def(implicit_init<U1Symmetry, U1>())
                        .def("__repr__",
                             [=](const U1Symmetry& symmetry) { return std::string("U1Symmetry") + "[" + std::to_string(symmetry.u1) + "]"; })
                        .def("__str__",
                             [=](const U1Symmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("u1", &U1Symmetry::u1),
            py::arg("FermiSymmmetry") =
                  py::class_<FermiSymmetry>(m, "FermiSymmetry")
                        .def(implicit_init<FermiSymmetry, Fermi>())
                        .def("__repr__",
                             [=](const FermiSymmetry& symmetry) { return std::string("FermiSymmetry") + "[" + std::to_string(symmetry.fermi) + "]"; })
                        .def("__str__",
                             [=](const FermiSymmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("fermi", &FermiSymmetry::fermi),
            py::arg("FermiZ2Symmmetry") = py::class_<FermiZ2Symmetry>(m, "FermiZ2Symmetry")
                                                .def(py::init<Fermi, Z2>())
                                                .def(implicit_init<FermiZ2Symmetry, const std::pair<Fermi, Z2>&>([](const std::pair<Fermi, Z2>& p) {
                                                   return std::make_unique<FermiZ2Symmetry>(p.first, p.second);
                                                }))
                                                .def("__repr__",
                                                     [=](const FermiZ2Symmetry& symmetry) {
                                                        return std::string("FermiZ2Symmetry") + "[" + std::to_string(symmetry.fermi) + "," +
                                                               std::to_string(symmetry.z2) + "]";
                                                     })
                                                .def("__str__",
                                                     [=](const FermiZ2Symmetry& symmetry) {
                                                        auto out = std::stringstream();
                                                        out << symmetry;
                                                        return out.str();
                                                     })
                                                .def_readwrite("fermi", &FermiZ2Symmetry::fermi)
                                                .def_readwrite("z2", &FermiZ2Symmetry::z2),
            py::arg("FermiU1Symmmetry") = py::class_<FermiU1Symmetry>(m, "FermiU1Symmetry")
                                                .def(py::init<Fermi, U1>())
                                                .def(implicit_init<FermiU1Symmetry, const std::pair<Fermi, U1>&>([](const std::pair<Fermi, U1>& p) {
                                                   return std::make_unique<FermiU1Symmetry>(p.first, p.second);
                                                }))
                                                .def("__repr__",
                                                     [=](const FermiU1Symmetry& symmetry) {
                                                        return std::string("FermiU1Symmetry") + "[" + std::to_string(symmetry.fermi) + "," +
                                                               std::to_string(symmetry.u1) + "]";
                                                     })
                                                .def("__str__",
                                                     [=](const FermiU1Symmetry& symmetry) {
                                                        auto out = std::stringstream();
                                                        out << symmetry;
                                                        return out.str();
                                                     })
                                                .def_readwrite("fermi", &FermiU1Symmetry::fermi)
                                                .def_readwrite("u1", &FermiU1Symmetry::u1));
      // edge
      m.attr("Edge") = py::dict(
#define DECLARE_EDGE(S) py::arg(#S) = declare_edge<S>(m, "Edge" #S)
            DECLARE_EDGE(NoSymmetry),
            DECLARE_EDGE(Z2Symmetry),
            DECLARE_EDGE(U1Symmetry),
            DECLARE_EDGE(FermiSymmetry),
            DECLARE_EDGE(FermiZ2Symmetry),
            DECLARE_EDGE(FermiU1Symmetry)
#undef DECLARE_EDGE
      );
      // tensor
      declare_tensor<double, NoSymmetry>(m, "TensorDoubleNoSymmetry");
      declare_tensor<double, U1Symmetry>(m, "TensorDoubleU1Symmetry");
      // TODO: python support
   }
} // namespace TAT
