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

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Configuration {
   int L1, L2;
   std::vector<std::optional<int>> data;

   Configuration(int L1, int L2) : L1(L1), L2(L2) {
      data.resize(L1 * L2);
   }

   void setitem(const std::tuple<int, int, int>& key, const std::optional<int>& value) {
      const auto& [l1, l2, orbit] = key;
      auto offset = (orbit * L1 + l1) * L2 + l2;
      while (offset >= data.size()) {
         data.resize(data.size() * 2);
      }
      data[offset] = value;
   }

   std::optional<int> getitem(const std::tuple<int, int, int>& key) const {
      const auto& [l1, l2, orbit] = key;
      auto offset = (orbit * L1 + l1) * L2 + l2;
      if (offset >= data.size()) {
         return {};
      } else {
         return data[offset];
      }
   }
};

auto dealing_configuration(py::module_& m) {
   using C = Configuration;
   return py::class_<C>(m, "Configuration", "Configuration for a state on square lattice")
         .def(py::init<int, int>(), "Create an empty configuration from an abstract state.", py::arg("L1"), py::arg("L2"))
         .def("__getitem__", &C::getitem)
         .def("__setitem__", &C::setitem)
         .def(
               "copy",
               [](const C& self) {
                  auto result = C(self.L1, self.L2);
                  result.data = self.data;
                  return result;
               },
               "Copy the configuration")
         .def_static(
               "export_orbit0",
               [](const std::vector<const C*>& configurations) {
                  auto configuration_number = configurations.size();
                  const auto& config0 = *configurations[0];
                  auto size = config0.L1 * config0.L2;
                  auto result = py::array_t<int>({Py_ssize_t(configuration_number), Py_ssize_t(1), Py_ssize_t(config0.L1), Py_ssize_t(config0.L2)});
                  auto pointer = static_cast<int*>(result.request().ptr);
                  for (auto c = 0; c < configuration_number; c++) {
                     for (auto i = 0; i < size; i++) {
                        pointer[c * size + i] = configurations[c]->data[i].value();
                     }
                  }
                  return result;
               },
               "Export configuration of orbit 0 as an array")
         .def_static(
               "get_hat",
               [](const std::vector<const C*>& configurations,
                  const std::vector<std::tuple<int, int, int>>& sites,
                  const std::vector<int>& physics_dims) {
                  auto configuration_number = configurations.size();
                  auto physics_edge_number = sites.size();
                  auto total_physics_dim = 1;
                  for (auto d : physics_dims) {
                     total_physics_dim *= d;
                  }
                  auto hat = py::array_t<int>({Py_ssize_t(configuration_number), Py_ssize_t(total_physics_dim)});
                  auto pointer = static_cast<int*>(hat.request().ptr);
                  std::fill(pointer, pointer + configuration_number * total_physics_dim, 0);
                  for (auto c = 0; c < configuration_number; c++) {
                     auto p = 0;
                     for (auto i = 0; i < physics_edge_number; i++) {
                        p *= physics_dims[i];
                        p += configurations[c]->getitem(sites[i]).value();
                     }
                     pointer[c * total_physics_dim + p] = 1;
                  }
                  return hat;
               });
}

#ifdef TAT_TETRAUX_SCALAPACK
const int zero = 0;
const int one = 1;
const int neg_one = -1;
#define SCALAPACK_ARRAY(X) (X).data, &one, &one, &(X)

#include "scalapack_functions.hpp"

struct blacs_context {
   int rank;
   int size;
   int ictxt;
   char layout;
   int nprow;
   int npcol;
   int myrow;
   int mycol;
   bool valid;
   blacs_context(char _layout, int _nprow, int _npcol) {
      layout = _layout;
      nprow = _nprow;
      npcol = _npcol;

      blacs_pinfo_(&rank, &size);
      blacs_get_(&neg_one, &zero, &ictxt);
      if (nprow == -1) {
         nprow = size / npcol;
      } else if (npcol == -1) {
         npcol = size / nprow;
      }
      blacs_gridinit_(&ictxt, &layout, &nprow, &npcol);
      blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
      valid = rank < nprow * npcol;
   }
   ~blacs_context() {
      if (valid) {
         blacs_gridexit_(&ictxt);
      }
   }
};

struct array_desc {
   int dtype; // 1 for dense matrix
   int ctxt;  // context
   int m;     // total size
   int n;     // total size
   int mb;    // block size
   int nb;    // block size
   int rsrc;
   int csrc;
   int lld;

   int local_m;
   int local_n;
   array_desc(const blacs_context& context, int _m, int _n, int _mb, int _nb) {
      dtype = 1;
      ctxt = context.ictxt;
      m = _m;
      n = _n;
      mb = _mb;
      nb = _nb;
      rsrc = zero;
      csrc = zero;

      local_m = numroc_(&m, &mb, &context.myrow, &zero, &context.nprow);
      local_n = numroc_(&n, &nb, &context.mycol, &zero, &context.npcol);

      lld = local_m; // fortran style
   }
};

template<typename ScalarType>
struct scalapack_array : array_desc {
   bool owndata;
   ScalarType* data;
   scalapack_array(const blacs_context& context, int _m, int _n, int _mb, int _nb, ScalarType* _data = nullptr) :
         array_desc(context, _m, _n, _mb, _nb) {
      if (_data) {
         owndata = false;
         data = _data;
      } else {
         owndata = true;
         data = new std::remove_const_t<ScalarType>[local_m * local_n];
      }
   }
   ~scalapack_array() {
      if (owndata) {
         delete[] data;
      }
   }
};

template<typename ScalarType>
auto pseudo_inverse(
      py::array_t<ScalarType, py::array::f_style>& delta,
      py::array_t<ScalarType>& energy,
      double r_pinv,
      double a_pinv,
      int total_n_s) {
   int info; // receive info from lapack/blas return code
   const ScalarType f_zero = 0;
   const ScalarType f_one = 1;

   int n_s = delta.shape(0);
   int n_p = delta.shape(1);

   auto context = blacs_context('C', -1, 1);

   auto Delta = scalapack_array<const ScalarType>(context, total_n_s, n_p, 1, n_p, delta.data(0, 0));
   if (Delta.local_m != n_s) {
      throw std::runtime_error("Mismatch between local sampling number and global sampling number.");
   }
   auto Energy = scalapack_array<const ScalarType>(context, total_n_s, 1, 1, 1, energy.data(0));

   auto T = scalapack_array<ScalarType>(context, total_n_s, total_n_s, 1, total_n_s);
   pgemm<ScalarType>("N", "C", &total_n_s, &total_n_s, &n_p, &f_one, SCALAPACK_ARRAY(Delta), SCALAPACK_ARRAY(Delta), &f_zero, SCALAPACK_ARRAY(T));

   // TODO opt
   // The process of this function is
   // Delta -> T -> U -> UE -> LUE -> ULUE -> Delta ULUE
   // where the step `T -> U` must be in context_square
   // Currently all other steps is in the default context.
   // I do not know whether it is the best.
   // And as for context_square, the block size is 4,
   // which may not be the best one neither.
   auto sqrt_size = int(std::pow(context.size, 1. / 2));
   auto context_square = blacs_context('C', sqrt_size, sqrt_size);
   auto T_square = scalapack_array<ScalarType>(context_square, total_n_s, total_n_s, 4, 4); // 4 * 4 block
   pgemr2d<ScalarType>(&total_n_s, &total_n_s, SCALAPACK_ARRAY(T), SCALAPACK_ARRAY(T_square), &context.ictxt);

   auto U_square = scalapack_array<ScalarType>(context_square, total_n_s, total_n_s, 4, 4);
   auto L = std::vector<real_scalar<ScalarType>>(total_n_s);
   if (context_square.valid) {
      auto info = psyevd<ScalarType>("V", "U", &total_n_s, SCALAPACK_ARRAY(T_square), L.data(), SCALAPACK_ARRAY(U_square));
      if (info != 0) {
         throw std::runtime_error("Error in P?SYEVD or P?HEEVD");
      }
   }
   if (context.rank == 0) {
      // Send L
      gebs2d<real_scalar<ScalarType>>(&context.ictxt, "All", " ", &total_n_s, &one, L.data(), &total_n_s);
   } else {
      // Recv L
      gebr2d<real_scalar<ScalarType>>(&context.ictxt, "All", " ", &total_n_s, &one, L.data(), &total_n_s, &zero, &zero);
   }

   auto U = scalapack_array<ScalarType>(context, total_n_s, total_n_s, 1, total_n_s);
   pgemr2d<ScalarType>(&total_n_s, &total_n_s, SCALAPACK_ARRAY(U_square), SCALAPACK_ARRAY(U), &context.ictxt);

   auto tmp1 = scalapack_array<ScalarType>(context, total_n_s, 1, 1, 1);
   pgemv<ScalarType>("C", &total_n_s, &total_n_s, &f_one, SCALAPACK_ARRAY(U), SCALAPACK_ARRAY(Energy), &one, &f_zero, SCALAPACK_ARRAY(tmp1), &one);

   // L is global
   auto L_max = L[iamax<real_scalar<ScalarType>>(&total_n_s, L.data(), &one)];
   auto num = r_pinv * L_max + a_pinv;
   for (auto i = 0; i < n_s; i++) {
      auto l = L[context.size * i + context.rank];
      tmp1.data[i] /= l * (1 + std::pow(num / l, 6));
   }

   auto tmp2 = scalapack_array<ScalarType>(context, total_n_s, 1, 1, 1);
   pgemv<ScalarType>("N", &total_n_s, &total_n_s, &f_one, SCALAPACK_ARRAY(U), SCALAPACK_ARRAY(tmp1), &one, &f_zero, SCALAPACK_ARRAY(tmp2), &one);

   // The last step does not use pgemv but gemv, and then reduce it in python
   auto result = py::array_t<ScalarType>(std::vector<Py_ssize_t>{n_p});
   auto result_p = static_cast<ScalarType*>(result.request().ptr);
   gemv<ScalarType>("C", &n_s, &n_p, &f_one, Delta.data, &Delta.lld, tmp2.data, &one, &f_zero, result_p, &one);
   return result;
}

auto dealing_pseudo_inverse(py::module_& m) {
   m.attr("pseudo_inverse_kernel_support") = true;
   m.def("pseudo_inverse_kernel",
         pseudo_inverse<float>,
         py::arg("delta"),
         py::arg("energy"),
         py::arg("r_pinv"),
         py::arg("a_pinv"),
         py::arg("total_n_s"));
   m.def("pseudo_inverse_kernel",
         pseudo_inverse<double>,
         py::arg("delta"),
         py::arg("energy"),
         py::arg("r_pinv"),
         py::arg("a_pinv"),
         py::arg("total_n_s"));
   m.def("pseudo_inverse_kernel",
         pseudo_inverse<std::complex<float>>,
         py::arg("delta"),
         py::arg("energy"),
         py::arg("r_pinv"),
         py::arg("a_pinv"),
         py::arg("total_n_s"));
   m.def("pseudo_inverse_kernel",
         pseudo_inverse<std::complex<double>>,
         py::arg("delta"),
         py::arg("energy"),
         py::arg("r_pinv"),
         py::arg("a_pinv"),
         py::arg("total_n_s"));
}
#else
auto dealing_pseudo_inverse(py::module_& m) {
   m.attr("pseudo_inverse_kernel_support") = false;
}
#endif

PYBIND11_MODULE(tetraux, m) {
   m.doc() = "tetraux contains some auxiliary function and class used by tetragono.";
   dealing_configuration(m);
   dealing_pseudo_inverse(m);
}
