export module TAT.const_integral;

import <variant>;

namespace TAT {
   /**
    * Maybe const integer
    *
    * Used in the situation where optmization can be applied for some small number
    *
    * \see to_const_integral
    */
   template<auto, typename DynamicType = void>
   struct const_integral_t {
      using value_type = DynamicType;
      value_type m_value;
      const_integral_t() = delete;
      const_integral_t(value_type v) : m_value(v) {}
      value_type value() const {
         return m_value;
      }
      static constexpr bool is_static = false;
      static constexpr bool is_dynamic = true;
   };

   template<auto StaticValue>
   struct const_integral_t<StaticValue, void> {
      using value_type = decltype(StaticValue);
      const_integral_t() = default;
      static constexpr value_type value() {
         return StaticValue;
      }
      static constexpr bool is_static = true;
      static constexpr bool is_dynamic = false;
   };

   template<typename T>
   const_integral_t(T) -> const_integral_t<0, T>;

   template<typename R, typename T>
   R to_const_integral_helper(T value) {
      return const_integral_t(value);
   }
   template<typename R, typename T, T first_value, T... possible_value>
   R to_const_integral_helper(T value) {
      if (first_value == value) {
         return const_integral_t<first_value>();
      } else {
         return to_const_integral_helper<R, T, possible_value...>(value);
      }
   }

   /**
    * Convert a variable to std::variant<run time variable, compile time variable 1, compile time variable 2, ...>
    *
    * Use std::visit([](const auto& variable) { xxx; variable.value(); xxx}, v) to dispatch compile time variable and run time variable
    */
   export template<typename T, T... possible_value>
   auto to_const_integral(T value) {
      using result_type = std::variant<const_integral_t<0, T>, const_integral_t<possible_value>...>;
      return to_const_integral_helper<result_type, T, possible_value...>(value);
   }

   export template<typename T>
   auto to_const_integral_0_to_16(T value) {
      return to_const_integral<T, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(value);
   }
} // namespace TAT
