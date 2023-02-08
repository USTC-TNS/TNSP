export module TAT.scalar_traits;

import <complex>;
import <type_traits>;

namespace TAT {
   // traits about scalar
   export template<typename T>
   concept is_real = std::is_scalar_v<T>;

   template<typename T>
   struct is_complex_helper : std::false_type {};
   template<typename T>
   struct is_complex_helper<std::complex<T>> : std::true_type {};
   export template<typename T>
   concept is_complex = is_complex_helper<T>::value;

   export template<typename T>
   concept is_scalar = is_real<T> || is_complex<T>;

   template<typename T>
   struct real_scalar_helper : std::conditional<is_real<T>, T, void> {};
   template<typename T>
   struct real_scalar_helper<std::complex<T>> : std::conditional<is_real<T>, T, void> {};
   /**
    * Get corresponding real type, used in svd and norm
    *
    * \tparam T if T is complex type, return corresponding basic real type, if it is real type, return itself, otherwise, return void
    */
   export template<typename T>
   using real_scalar = real_scalar_helper<T>::type;
} // namespace TAT
