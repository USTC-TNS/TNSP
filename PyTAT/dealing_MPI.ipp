#include "PyTAT.hpp"
namespace TAT {
   void FUNCTION_NAME(py::class_<mpi_t>& py_mpi_t) {
      using Scalar = SCALAR_NAME;
      using Symmetry = SYMMETRY_NAME;
#ifdef TAT_USE_MPI
      py_mpi_t.def_static("send", &mpi_t::send<Tensor<Scalar, Symmetry>>)
            .def_static("receive", &mpi_t::receive<Tensor<Scalar, Symmetry>>)
            .def("send_receive", &mpi_t::send_receive<Tensor<Scalar, Symmetry>>)
            .def("broadcast", &mpi_t::broadcast<Tensor<Scalar, Symmetry>>)
            .def("reduce",
                 [](const mpi_t& self,
                    const Tensor<Scalar, Symmetry>& value,
                    const int root,
                    std::function<Tensor<Scalar, Symmetry>(const Tensor<Scalar, Symmetry>&, const Tensor<Scalar, Symmetry>&)> func) {
                    return self.reduce(value, root, func);
                 })
            .def("summary", [](const mpi_t& self, const Tensor<Scalar, Symmetry>& value, const int root) {
               return self.reduce(value, root, [](const Tensor<Scalar, Symmetry>& a, const Tensor<Scalar, Symmetry>& b) { return a + b; });
            });
#endif
   }
} // namespace TAT
