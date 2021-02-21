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
   py::class_<mpi_t> set_mpi(py::module_& tat_m) {
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
      return py_mpi_t;
   }
} // namespace TAT
