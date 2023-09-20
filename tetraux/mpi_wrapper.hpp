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

#ifndef MPI_WRAPPER_HPP
#define MPI_WRAPPER_HPP

#include <complex>

template<typename Scalar>
MPI_Datatype mpi_datatype;

template<>
inline MPI_Datatype mpi_datatype<float> = MPI_FLOAT;
template<>
inline MPI_Datatype mpi_datatype<double> = MPI_DOUBLE;
template<>
inline MPI_Datatype mpi_datatype<std::complex<float>> = MPI_COMPLEX;
template<>
inline MPI_Datatype mpi_datatype<std::complex<double>> = MPI_DOUBLE_COMPLEX;

#endif
