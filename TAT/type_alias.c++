module;

#include <cstdint>

export module TAT.type_alias;

namespace TAT {
   // type alias

   // The most common used integral type `short`, `int`, `long` have different size in different platform
   // in linux, they are 16, 32, 64(lp64)
   // in windows, they are 16, 32, 32(llp64)
   // So use uintxx_t explicitly to avoid it incompatible when import data exported in another platform
   // In TAT, there is also `int` type common used, especially when calling blas or lapack function
   // It is all 32 bit in most platform currently.
   // But it is 64bit in ilp64 and 16bit in lp32
   // So please do not link blas lapack using ilp64 or lp32
   /**
    * Tensor rank type
    */
   export using Rank = std::uint16_t;
   /**
    * Tensor block number, or dimension segment number type
    */
   export using Nums = std::uint32_t;
   /**
    * Tensor content data size, or dimension size type
    */
   export using Size = std::uint64_t;

   /**
    * Fermi arrow type
    *
    * \note For connected two edge, EPR pair is \f$a^\dagger b^\dagger\f$
    * then, tensor owning edge a have arrow=false, and tensor owning edge b has arrow=true.
    * namely, the EPR pair arrow order is (false true)
    */
   export using Arrow = bool;
} // namespace TAT
