#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "TAT/TAT.hpp"

namespace TAT {
   namespace py = pybind11;

   template<class Symmetry>
   auto declare_edge(py::module& m, const char* name) {
      auto result = py::class_<Edge<Symmetry>>(m, name)
                          .def_readwrite("map", &Edge<Symmetry>::map)
                          .def(py::init<Size>())
                          .def(py::init<std::map<Symmetry, Size>>())
                          .def(py::init<const std::set<Symmetry>&>())
                          .def("__str__", [](const Edge<Symmetry>& edge) {
                             auto out = std::stringstream();
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
            .def(py::init<const std::string&>())
            .def("__repr__", [](const Name& name) { return "Name[" + id_to_name.at(name.id) + "]"; })
            .def("__str__", [](const Name& name) { return id_to_name.at(name.id); })
            .def_readonly("id", &Name::id)
            .def_property_readonly("name", [](const Name& name) { return id_to_name.at(name.id); });

      py::implicitly_convertible<std::string, Name>();
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
                        .def(py::init<Z2>())
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
                        .def(py::init<U1>())
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
                        .def(py::init<Fermi>())
                        .def("__repr__",
                             [=](const FermiSymmetry& symmetry) { return std::string("FermiSymmetry") + "[" + std::to_string(symmetry.fermi) + "]"; })
                        .def("__str__",
                             [=](const FermiSymmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("fermi", &FermiSymmetry::fermi),
            py::arg("FermiZ2Symmmetry") =
                  py::class_<FermiZ2Symmetry>(m, "FermiZ2Symmetry")
                        .def(py::init<Fermi, Z2>())
                        .def("__repr__",
                             [=](const FermiZ2Symmetry& symmetry) {
                                return std::string("FermiZ2Symmetry") + "[" + std::to_string(symmetry.fermi) + "," + std::to_string(symmetry.z2) +
                                       "]";
                             })
                        .def("__str__",
                             [=](const FermiZ2Symmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("fermi", &FermiZ2Symmetry::fermi)
                        .def_readwrite("z2", &FermiZ2Symmetry::z2)
                        .def(py::init([](const std::pair<Fermi, Z2>& arr) { return std::make_unique<FermiZ2Symmetry>(arr.first, arr.second); })),
            py::arg("FermiU1Symmmetry") =
                  py::class_<FermiU1Symmetry>(m, "FermiU1Symmetry")
                        .def(py::init<Fermi, U1>())
                        .def("__repr__",
                             [=](const FermiU1Symmetry& symmetry) {
                                return std::string("FermiU1Symmetry") + "[" + std::to_string(symmetry.fermi) + "," + std::to_string(symmetry.u1) +
                                       "]";
                             })
                        .def("__str__",
                             [=](const FermiU1Symmetry& symmetry) {
                                auto out = std::stringstream();
                                out << symmetry;
                                return out.str();
                             })
                        .def_readwrite("fermi", &FermiU1Symmetry::fermi)
                        .def_readwrite("u1", &FermiU1Symmetry::u1)
                        .def(py::init([](const std::pair<Fermi, U1>& arr) { return std::make_unique<FermiU1Symmetry>(arr.first, arr.second); })));

      py::implicitly_convertible<Z2, Z2Symmetry>();
      py::implicitly_convertible<U1, U1Symmetry>();
      py::implicitly_convertible<Fermi, FermiSymmetry>();
      py::implicitly_convertible<std::pair<Fermi, Z2>, FermiZ2Symmetry>();
      py::implicitly_convertible<std::pair<Fermi, U1>, FermiU1Symmetry>();
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
      // TODO: python support
   }
} // namespace TAT
