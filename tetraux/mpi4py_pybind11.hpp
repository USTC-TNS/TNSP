/**
 * Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef MPI4PY_PYBIND11_HPP
#define MPI4PY_PYBIND11_HPP

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <mpi4py/mpi4py.h>

// https://stackoverflow.com/a/62449190/7832640

struct mpi4py_comm {
    mpi4py_comm() : value(MPI_COMM_NULL) { }
    mpi4py_comm(MPI_Comm value) : value(value) { }
    operator MPI_Comm() {
        return value;
    }

    MPI_Comm value;
};

namespace pybind11 {
    namespace detail {
        template<>
        struct type_caster<mpi4py_comm> {
          public:
            PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

            // Python -> C++
            bool load(handle src, bool) {
                PyObject* py_src = src.ptr();

                // Check that we have been passed an mpi4py communicator
                if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
                    // Convert to regular MPI communicator
                    value.value = *PyMPIComm_Get(py_src);
                } else {
                    return false;
                }

                return !PyErr_Occurred();
            }

            // C++ -> Python
            static handle cast(mpi4py_comm src, return_value_policy /* policy */, handle /* parent */) {
                // Create an mpi4py handle
                return PyMPIComm_New(src.value);
            }
        };
    } // namespace detail
} // namespace pybind11
#endif
