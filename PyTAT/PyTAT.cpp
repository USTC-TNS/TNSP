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

#include "PyTAT.hpp"

namespace TAT {

    bool parity_and_arrow(int p) {
        if (p == +1) {
            return false;
        } else if (p == -1) {
            return true;
        } else {
            throw std::runtime_error("Parity should be either +1 or -1.");
        }
    }

#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
    std::function<void()> dealing_Tensor_##SYM##_##SCALARSHORT( \
        py::module_& symmetry_m, \
        const std::string& scalar_short_name, \
        const std::string& scalar_name, \
        const std::string& symmetry_short_name \
    );
    TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

    PYBIND11_MODULE(TAT, tat_m) {
        tat_m.doc() = "TAT is A Tensor library!";
        tat_m.attr("__version__") = version;
        tat_m.attr("version") = version;
        tat_m.attr("information") = information;
        // callable
        set_callable(tat_m);
        // random
        set_random(tat_m);

        // symmetry and edge and edge segment
        auto No_m = tat_m.def_submodule("No", "A submodule contains non-symmetry tensors");
        dealing_symmetry<NoSymmetry>(No_m, "No").def(py::init<>()).def(implicit_from_tuple<NoSymmetry, std::tuple<>>(), py::arg("empty_tuple"));
        dealing_edge<NoSymmetry, false>(No_m, "No");
        dealing_edge<NoSymmetry, true>(No_m, "No");

        auto Z2_m = tat_m.def_submodule("Z2", "A submodule contains Z2 symmetry tensors");
        dealing_symmetry<Z2Symmetry>(Z2_m, "Z2")
            .def(py::init<>())
            .def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2"))
            .def(implicit_from_tuple<Z2Symmetry, std::tuple<Z2>>(), py::arg("tuple_of_z2"))
            .def_property_readonly("z2", [](const Z2Symmetry& symmetry) { return std::get<0>(symmetry); });
        dealing_edge<Z2Symmetry, false>(Z2_m, "Z2");
        dealing_edge<Z2Symmetry, true>(Z2_m, "Z2");

        auto U1_m = tat_m.def_submodule("U1", "A submodule contains U1 symmetry tensors");
        dealing_symmetry<U1Symmetry>(U1_m, "U1")
            .def(py::init<>())
            .def(implicit_init<U1Symmetry, U1>(), py::arg("u1"))
            .def(implicit_from_tuple<U1Symmetry, std::tuple<U1>>(), py::arg("tuple_of_u1"))
            .def_property_readonly("u1", [](const U1Symmetry& symmetry) { return std::get<0>(symmetry); });
        dealing_edge<U1Symmetry, false>(U1_m, "U1");
        dealing_edge<U1Symmetry, true>(U1_m, "U1");

        auto Fermi_m = tat_m.def_submodule("Fermi", "A submodule contains fermion U1 symmetry tensors");
        dealing_symmetry<FermiSymmetry>(Fermi_m, "Fermi")
            .def(py::init<>())
            .def(implicit_init<FermiSymmetry, U1>(), py::arg("fermi"))
            .def(implicit_from_tuple<FermiSymmetry, std::tuple<U1>>(), py::arg("tuple_of_fermi"))
            .def_property_readonly("fermi", [](const FermiSymmetry& symmetry) { return std::get<0>(symmetry); });
        dealing_edge<FermiSymmetry, false>(Fermi_m, "Fermi");
        dealing_edge<FermiSymmetry, true>(Fermi_m, "Fermi");

        auto FermiZ2_m = tat_m.def_submodule("FermiZ2", "A submodule contains fermion U1 cross Z2 symmetry tensors");
        dealing_symmetry<FermiZ2Symmetry>(FermiZ2_m, "FermiZ2")
            .def(py::init<>())
            .def(py::init<U1, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_from_tuple<FermiZ2Symmetry, std::tuple<U1, Z2>>(), py::arg("tuple_of_fermi_z2"))
            .def_property_readonly("fermi", [](const FermiZ2Symmetry& symmetry) { return std::get<0>(symmetry); })
            .def_property_readonly("z2", [](const FermiZ2Symmetry& symmetry) { return std::get<1>(symmetry); });
        dealing_edge<FermiZ2Symmetry, false>(FermiZ2_m, "FermiZ2");
        dealing_edge<FermiZ2Symmetry, true>(FermiZ2_m, "FermiZ2");

        auto FermiU1_m = tat_m.def_submodule("FermiU1", "A submodule contains fermion U1 cross U1 symmetry tensors");
        dealing_symmetry<FermiU1Symmetry>(FermiU1_m, "FermiU1")
            .def(py::init<>())
            .def(py::init<U1, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_from_tuple<FermiU1Symmetry, std::tuple<U1, U1>>(), py::arg("tuple_of_fermi_u1"))
            .def_property_readonly("fermi", [](const FermiU1Symmetry& symmetry) { return std::get<0>(symmetry); })
            .def_property_readonly("u1", [](const FermiU1Symmetry& symmetry) { return std::get<1>(symmetry); });
        dealing_edge<FermiU1Symmetry, false>(FermiU1_m, "FermiU1");
        dealing_edge<FermiU1Symmetry, true>(FermiU1_m, "FermiU1");

        auto Parity_m = tat_m.def_submodule("Parity", "A submodule contains fermion Z2 symmetry tensors");
        dealing_symmetry<ParitySymmetry>(Parity_m, "Parity")
            .def(py::init<>())
            .def(implicit_init<ParitySymmetry, Z2>(), py::arg("parity"))
            .def(implicit_from_tuple<ParitySymmetry, std::tuple<Z2>>(), py::arg("tuple_of_z2"))
            .def_property_readonly("parity", [](const ParitySymmetry& symmetry) { return std::get<0>(symmetry); });
        dealing_edge<ParitySymmetry, false>(Parity_m, "Parity");
        dealing_edge<ParitySymmetry, true>(Parity_m, "Parity");

        auto FermiFermi_m = tat_m.def_submodule("FermiFermi", "A submodule contains fermion U1 cross fermion U1 symmetry tensors");
        dealing_symmetry<FermiFermiSymmetry>(FermiFermi_m, "FermiFermi")
            .def(py::init<>())
            .def(py::init<U1, U1>(), py::arg("fermi_0"), py::arg("fermi_1"))
            .def(implicit_from_tuple<FermiFermiSymmetry, std::tuple<U1, U1>>(), py::arg("tuple_of_fermi_0_fermi_1"))
            .def_property_readonly("fermi_0", [](const FermiFermiSymmetry& symmetry) { return std::get<0>(symmetry); })
            .def_property_readonly("fermi_1", [](const FermiFermiSymmetry& symmetry) { return std::get<1>(symmetry); });
        dealing_edge<FermiFermiSymmetry, false>(FermiFermi_m, "FermiFermi");
        dealing_edge<FermiFermiSymmetry, true>(FermiFermi_m, "FermiFermi");

        // tensor
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) at_exit(dealing_Tensor_##SYM##_##SCALARSHORT(SYM##_m, #SCALARSHORT, #SCALAR, #SYM));
        TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

        // Set alias
        tat_m.attr("Normal") = tat_m.attr("No");
        for (auto sym_m_name : std::vector{"No", "Z2", "U1", "Fermi", "FermiZ2", "FermiU1", "Parity", "FermiFermi"}) {
            auto&& sym_m = tat_m.attr(sym_m_name);

            sym_m.attr("float") = sym_m.attr("D");
            sym_m.attr("complex") = sym_m.attr("Z");

            sym_m.attr("float32") = sym_m.attr("S");
            sym_m.attr("float64") = sym_m.attr("D");
            sym_m.attr("complex64") = sym_m.attr("C");
            sym_m.attr("complex128") = sym_m.attr("Z");
        }

        // Shaojun Dong wants to use +1 and -1 to identify parity instead of False and True, make him happy.
        tat_m.def("parity", parity_and_arrow, "A compatibility function for converting Z2 symmetry from sjdong's convention to TAT's convention");
        tat_m.def("arrow", parity_and_arrow, "A compatibility function for converting fermi-arrow from sjdong's convention to TAT's convention");

        at_exit.release();
    }
} // namespace TAT
