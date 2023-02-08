module;

#include <cstdint>

export module TAT.hash_absorb;

namespace TAT {
   export std::size_t& hash_absorb(std::size_t& seed, std::size_t value) {
      // copy from boost
      return seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   }
} // namespace TAT
