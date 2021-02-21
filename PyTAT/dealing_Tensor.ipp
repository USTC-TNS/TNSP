#include "PyTAT.hpp"
namespace TAT {
   void FUNCTION_NAME(
         py::module_& symmetry_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      using Scalar = SCALAR_NAME;
      using Symmetry = SYMMETRY_NAME;
      declare_tensor<Scalar, Symmetry>(symmetry_m, scalar_short_name, scalar_name, symmetry_short_name);
   }
} // namespace TAT
