/**
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#define TAT_USE_SIMPLE_NAME
#include <TAT/TAT.hpp>

#include "run_test.hpp"

void run_test() {
   do {
      auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {2, 3, 4, 5}}.range();
      std::cout << a << "\n";
      auto [u, s, v] = a.svd({"C", "A"}, "E", "F", "U", "V");
      std::cout << u << "\n";
      std::cout << v << "\n";
      std::cout << s << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "U"}}).edge_rename({{"V", "F"}}), u, {{"F", "E"}}).transpose({"A", "B", "C", "D"})
                << "\n";
   } while (false);
   do {
      auto b = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {2, 3, 4, 5}}.range();
      std::cout << b << "\n";
      auto [u, s, v] = b.svd({"A", "D"}, "E", "F", "U", "V");
      std::cout << u << "\n";
      std::cout << v << "\n";
      std::cout << s << "\n";
      std::cout << decltype(v)::contract(v, v.edge_rename({{"F", "F2"}}), {{"B", "B"}, {"C", "C"}}).transform([](auto i) {
         return std::abs(i) > 1e-5 ? i : 0;
      }) << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "V"}}).edge_rename({{"U", "F"}}), u, {{"F", "E"}}).transpose({"A", "B", "C", "D"})
                << "\n";
   } while (false);
#define t_edge(...) \
   { {__VA_ARGS__}, true }
#define f_edge(...) \
   { {__VA_ARGS__}, false }
   do {
      auto c =
            TAT::Tensor<double, TAT::U1Symmetry>{
                  {"A", "B", "C", "D"},
                  {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range();
      auto [u, s, v] = c.svd({"C", "A"}, "E", "F", "U", "V");
      std::cout << u << "\n";
      std::cout << s << "\n";
      std::cout << v << "\n";
      std::cout << c << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "V"}}).edge_rename({{"U", "F"}}), u, {{"F", "E"}})
                         .transpose({"A", "B", "C", "D"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
      std::cout << decltype(v)::contract(v, u.contract(s, {{"E", "U"}}).edge_rename({{"V", "E"}}), {{"F", "E"}})
                         .transpose({"A", "B", "C", "D"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
   } while (false);
   do {
      auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {2, 3, 4, 5}}.range();
      std::cout << a << "\n";
      auto [u, s, v] = a.svd({"C", "A"}, "E", "F", "U", "V", TAT::Size(2));
      std::cout << u << "\n";
      std::cout << v << "\n";
      std::cout << s << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "U"}}).edge_rename({{"V", "F"}}), u, {{"F", "E"}})
                         .transpose({"A", "B", "C", "D"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
   } while (false);
   do {
      auto c =
            TAT::Tensor<double, TAT::U1Symmetry>{
                  {"A", "B", "C", "D"},
                  {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range();
      auto [u, s, v] = c.svd({"C", "A"}, "E", "F", "U", "V", TAT::Size(7));
      std::cout << u << "\n";
      std::cout << s << "\n";
      std::cout << v << "\n";
      std::cout << c << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "V"}}).edge_rename({{"U", "F"}}), u, {{"F", "E"}})
                         .transpose({"A", "B", "C", "D"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
      std::cout << decltype(v)::contract(v, u.contract(s, {{"E", "U"}}).edge_rename({{"V", "E"}}), {{"F", "E"}})
                         .transpose({"A", "B", "C", "D"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
   } while (false);
   do {
      auto c =
            TAT::Tensor<double, TAT::FermiSymmetry>{
                  {"A", "B", "C", "D"},
                  {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range()
                  .transpose({"D", "C", "B", "A"});
      auto [u, s, v] = c.svd({"C", "A"}, "E", "F", "U", "V");
      std::cout << u << "\n";
      std::cout << s << "\n";
      std::cout << v << "\n";
      std::cout << c << "\n";
      std::cout << decltype(v)::contract(v.contract(s, {{"F", "V"}}).edge_rename({{"U", "F"}}), u, {{"F", "E"}})
                         .transpose({"D", "C", "B", "A"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
      std::cout << decltype(v)::contract(v, u.contract(s, {{"E", "U"}}).edge_rename({{"V", "E"}}), {{"F", "E"}})
                         .transpose({"D", "C", "B", "A"})
                         .transform([](auto i) {
                            return std::abs(i) < 0.01 ? 0 : i;
                         })
                << "\n";
   } while (false);
}