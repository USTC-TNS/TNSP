for Sym in No Z2 U1 Fermi FermiZ2 FermiU1
do
   for ShortScalar in "S float" "D double" "C std::complex<float>" "Z std::complex<double>"
   do
      Short=`awk '{print $1}' <<< $ShortScalar`
      Scalar=`awk '{print $2}' <<< $ShortScalar`
      cat << PYTAT_DEALING > dealing_Tensor_$Short$Sym.cpp
#include "PyTAT.hpp"
namespace TAT {
   void dealing_Tensor_${Short}${Sym}(
         py::module_& tensor_m,
         py::module_& block_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      using Scalar = ${Scalar};
      using Symmetry = ${Sym}Symmetry;
      declare_tensor<Scalar, Symmetry>(tensor_m, block_m, scalar_short_name, scalar_name, symmetry_short_name);
   }
} // namespace TAT
PYTAT_DEALING
      cat << PYTAT_DEALING > dealing_MPI_$Short$Sym.cpp
#include "PyTAT.hpp"
namespace TAT {
   void dealing_MPI_${Short}${Sym}(py::class_<mpi_t>& py_mpi_t) {
      using Scalar = ${Scalar};
      using Symmetry = ${Sym}Symmetry;
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
PYTAT_DEALING
   done
done
