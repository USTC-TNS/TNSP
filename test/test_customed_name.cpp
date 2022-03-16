/**
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <TAT/TAT.hpp>

#include "run_test.hpp"

namespace net {
   using pss = std::pair<std::string, std::string>;

   std::ostream& operator<<(std::ostream& os, const pss& p) {
      return os << std::get<0>(p) << "." << std::get<1>(p);
   }
} // namespace net

namespace std {
   template<>
   struct hash<net::pss> {
      size_t operator()(const net::pss& name) const {
         std::hash<std::string> string_hash;
         return string_hash(name.first) ^ !string_hash(name.second);
      }
   };
} // namespace std

namespace TAT {
   using net::pss;

   template<>
   const pss InternalName<pss>::Default_0 = {"Internal", "0"};
   template<>
   const pss InternalName<pss>::Default_1 = {"Internal", "1"};
   template<>
   const pss InternalName<pss>::Default_2 = {"Internal", "2"};

   template<>
   struct NameTraits<pss> {
      static constexpr name_out_operator_t<pss> print = net::operator<<;
   };
} // namespace TAT

namespace net {
   using T = ::TAT::Tensor<double, TAT::NoSymmetry, pss>;

   void f() {
      auto i0 = TAT::InternalName<pss>::SVD_U;
      TAT::NameTraits<pss>::print(std::cout, i0) << "\n";
      auto a = T({{"A", "1"}}, {5}).range();
      std::cout << a << "\n";

      auto s = a.svd({{"A", "1"}}, {"A", "U"}, {"A", "V"}, {"S", "U"}, {"S", "V"});
      std::cout << s.U << "\n";
      std::cout << s.S << "\n";
      std::cout << s.V << "\n";
      std::cout << s.V.edge_rename(std::map<pss, std::string>{{{"A", "V"}, "V"}}) << "\n";
   }
} // namespace net

void run_test() {
   net::f();
}
