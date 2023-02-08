module;

#include <array>

export module TAT.empty_list;

namespace TAT {
   export template<typename T>
   struct empty_list : std::array<T, 0> {
      auto find(const auto&) const {
         return this->end();
      }
   };
} // namespace TAT
