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

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "blas_lapack.hpp"
#include "mpi4py_pybind11.hpp"
#include "mpi_wrapper.hpp"

namespace py = pybind11;

template<typename Scalar>
auto cg(
    int Ns,
    int Np,
    const Scalar* Energy_h, // Ns
    const Scalar* Delta_h, // Ns * Np
    Scalar* x_h, // Np // x is result
    int step, // stop condition
    double error, // stop condition
    int device,
    MPI_Comm comm
) {
    gpu_handle handle(device);

    gpu_array<Scalar> Energy(Ns);
    Energy.from_host(Energy_h);
    gpu_array<Scalar> Delta(Ns * Np);
    Delta.from_host(Delta_h);
    gpu_array<Scalar> x(Np);

    // Initialize variables
    Scalar f_one(1);
    Scalar f_neg_one(-1);
    Scalar f_zero(0);
    int i_one(1);
    gpu_array<Scalar> b(Np);
    gpu_array<Scalar> r(Np);
    gpu_array<Scalar> p(Np);
    gpu_array<Scalar> Dp(Ns);
    gpu_array<Scalar> Ap(Np);
    real_base<Scalar> b_square;
    real_base<Scalar> r_square;
    real_base<Scalar> new_r_square;
    real_base<Scalar> alpha;
    real_base<Scalar> beta;
    std::unique_ptr<Scalar[]> res_h(new Scalar[Ns]);

    // For complex scalar, conjugate on Delta is needed
    if constexpr (!std::is_same_v<real_base<Scalar>, Scalar>) {
        real_base<Scalar> neg_one = -1;
        scal<real_base<Scalar>>(handle.get(), Ns * Np, &neg_one, reinterpret_cast<real_base<Scalar>*>(Delta.get()) + 1, 2);
    }

    // Auxiliary functions
    auto D = [&](gpu_array<Scalar>& v, // Np
                 gpu_array<Scalar>& res // Ns
             ) {
        // res = Delta @ v
        // Previous: N -> fix fortran order -> T -> manually conjugate -> C
        gemv<Scalar>(handle.get(), blas_op_c, Np, Ns, &f_one, Delta.get(), Np, v.get(), i_one, &f_zero, res.get(), i_one);
        // allreduce res
        // TODO: For rocm aware mpi, it is possible to allreduce directly inside gpu.
        res.to_host(res_h.get());
        MPI_Allreduce(MPI_IN_PLACE, res_h.get(), Ns, mpi_datatype<Scalar>, MPI_SUM, comm);
        res.from_host(res_h.get());
    };
    auto DT = [&](gpu_array<Scalar>& v, // Ns
                  gpu_array<Scalar>& res // Np
              ) {
        // res = Delta.H @ v
        // Previous: C -> fix fortran order -> C without T -> manually conjugate -> N
        gemv<Scalar>(handle.get(), blas_op_n, Np, Ns, &f_one, Delta.get(), Np, v.get(), i_one, &f_zero, res.get(), i_one);
    };

    // CG

    // b = D.H @ Energy
    DT(Energy, b);
    // b_square = b.H @ b
    nrm2<Scalar>(handle.get(), Np, b.get(), i_one, &b_square);
    b_square = b_square * b_square;
    MPI_Allreduce(MPI_IN_PLACE, &b_square, 1, mpi_datatype<real_base<Scalar>>, MPI_SUM, comm);

    // x = 0
    scal<Scalar>(handle.get(), Np, &f_zero, x.get(), i_one);
    // r = b
    copy<Scalar>(handle.get(), Np, b.get(), i_one, r.get(), i_one);
    // p = r
    copy<Scalar>(handle.get(), Np, r.get(), i_one, p.get(), i_one);
    // r_square = r.H @ r
    nrm2<Scalar>(handle.get(), Np, r.get(), i_one, &r_square);
    r_square = r_square * r_square;
    MPI_Allreduce(MPI_IN_PLACE, &r_square, 1, mpi_datatype<real_base<Scalar>>, MPI_SUM, comm);

    int t = 0;
    const char* reason;
    while (true) {
        if (t == step) {
            reason = "max step count reached";
            break;
        }
        if (error != 0.0 && error * error > r_square / b_square) {
            reason = "r^2 is small enough";
            break;
        }
        // show conjugate gradient step=t r^2/b^2=r_square/b_square
        // Dp = Delta @ p
        D(p, Dp);
        // alpha = r_square / allreduce(Dp.H @ Dp)
        nrm2<Scalar>(handle.get(), Ns, Dp.get(), i_one, &alpha);
        alpha = alpha * alpha;
        alpha = r_square / alpha;
        // x = x + alpha * p
        Scalar c_alpha = alpha;
        axpy<Scalar>(handle.get(), Np, &c_alpha, p.get(), i_one, x.get(), i_one);
        // r = r - alpha * D.H @ Dp
        DT(Dp, Ap);
        Scalar n_alpha = -alpha;
        axpy<Scalar>(handle.get(), Np, &n_alpha, Ap.get(), i_one, r.get(), i_one);
        // new_r_square = r.H @ r
        nrm2<Scalar>(handle.get(), Np, r.get(), i_one, &new_r_square);
        new_r_square = new_r_square * new_r_square;
        MPI_Allreduce(MPI_IN_PLACE, &new_r_square, 1, mpi_datatype<real_base<Scalar>>, MPI_SUM, comm);
        // beta = new_r_square / r_square
        beta = new_r_square / r_square;
        // r_square = new_r_square
        r_square = new_r_square;
        // p = r + beta * p
        Scalar c_beta = beta;
        scal<Scalar>(handle.get(), Np, &c_beta, p.get(), i_one);
        axpy<Scalar>(handle.get(), Np, &f_one, r.get(), i_one, p.get(), i_one);

        t += 1;
    }

    x.to_host(x_h);

    return std::make_tuple(reason, t, r_square / b_square);
}

template<typename Scalar>
std::tuple<py::array_t<Scalar>, std::string, int, double> py_cg(
    int Ns,
    int Np,
    py::array_t<Scalar, py::array::c_style> Energy,
    py::array_t<Scalar, py::array::c_style> Delta,
    int step,
    double error,
    int device,
    mpi4py_comm comm
) {
    auto result = py::array_t<Scalar>({Py_ssize_t(Np)});
    auto buf_e = Energy.request();
    auto buf_d = Delta.request();
    auto buf_r = result.request();
    if (buf_e.ndim != 1 || buf_e.shape[0] != Ns) {
        throw std::runtime_error("Energy vector shape mismatch");
    }
    if (buf_d.ndim != 2 || buf_d.shape[0] != Ns || buf_d.shape[1] != Np) {
        throw std::runtime_error("Delta matrix shape mismatch");
    }
    const auto& [reason, result_step, result_error] = cg<Scalar>(
        Ns,
        Np,
        static_cast<Scalar*>(buf_e.ptr),
        static_cast<Scalar*>(buf_d.ptr),
        static_cast<Scalar*>(buf_r.ptr),
        step,
        error,
        device,
        comm.value
    );
    return std::make_tuple(std::move(result), reason, result_step, result_error);
}

void dealing_cg(py::module_& m) {
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API.");
    }
    m.def(
        "cg",
        py_cg<float>,
        py::arg("Ns"),
        py::arg("Np"),
        py::arg("Energy"),
        py::arg("Delta"),
        py::arg("step"),
        py::arg("error"),
        py::arg("device"),
        py::arg("comm")
    );
    m.def(
        "cg",
        py_cg<double>,
        py::arg("Ns"),
        py::arg("Np"),
        py::arg("Energy"),
        py::arg("Delta"),
        py::arg("step"),
        py::arg("error"),
        py::arg("device"),
        py::arg("comm")
    );
    m.def(
        "cg",
        py_cg<std::complex<float>>,
        py::arg("Ns"),
        py::arg("Np"),
        py::arg("Energy"),
        py::arg("Delta"),
        py::arg("step"),
        py::arg("error"),
        py::arg("device"),
        py::arg("comm")
    );
    m.def(
        "cg",
        py_cg<std::complex<double>>,
        py::arg("Ns"),
        py::arg("Np"),
        py::arg("Energy"),
        py::arg("Delta"),
        py::arg("step"),
        py::arg("error"),
        py::arg("device"),
        py::arg("comm")
    );
}

PYBIND11_MODULE(tetraux, m) {
    dealing_cg(m);
}
