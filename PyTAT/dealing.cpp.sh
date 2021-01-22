for Sym in No Z2 U1 Fermi FermiZ2 FermiU1
do 
   for ShortScalar in "S float" "D double" "C std::complex<float>" "Z std::complex<double>"
   do
      Short=`awk '{print $1}' <<< $ShortScalar`
      Scalar=`awk '{print $2}' <<< $ShortScalar`
      cat << PYTAT_DEALING > dealing_$Short$Sym.cpp
#include "PyTAT.hpp"
namespace TAT {
   void dealing_${Short}${Sym}(
         py::module_& tensor_m,
         py::module_& block_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      declare_tensor<${Scalar}, ${Sym}Symmetry>(tensor_m, block_m, scalar_short_name, scalar_name, symmetry_short_name);
   }
} // namespace TAT
PYTAT_DEALING
   done
done
