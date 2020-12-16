/**
 * \file square_auxiliaries_system.hpp
 *
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#pragma once
#ifndef SQUARE_SQUARE_AUXILIARIES_SYSTEM_HPP
#define SQUARE_SQUARE_AUXILIARIES_SYSTEM_HPP

#include <lazy.hpp>

#include "basic.hpp"

namespace square {
   template<typename T>
   struct SquareAuxiliariesSystem {
      int M;
      int N;
      Size dimension_cut;
      lazy::Graph graph;
      std::vector<std::vector<std::shared_ptr<lazy::root<Tensor<T>>>>> lattice;

      std::map<int, std::shared_ptr<lazy::node<std::vector<Tensor<T>>>>> up_to_down, down_to_up, left_to_right, right_to_left;
      std::map<int, std::map<int, std::shared_ptr<lazy::node<Tensor<T>>>>> up_to_down_3_3, up_to_down_3_1, down_to_up_3_3, down_to_up_3_1,
            left_to_right_3_3, left_to_right_3_1, right_to_left_3_3, right_to_left_3_1;

      SquareAuxiliariesSystem() = default;

      SquareAuxiliariesSystem(const SquareAuxiliariesSystem<T>& other) : SquareAuxiliariesSystem(other.M, other.N, other.dimension_cut) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               lattice[i][j]->set(other.lattice[i][j].get());
            }
         }
      }
      SquareAuxiliariesSystem& operator=(const SquareAuxiliariesSystem& other) {
         if (this != &other) {
            new (this) SquareAuxiliariesSystem(other);
         }
         return this;
      }
      SquareAuxiliariesSystem(SquareAuxiliariesSystem<T>&& other) = default;
      SquareAuxiliariesSystem& operator=(SquareAuxiliariesSystem&& other) = default;

      SquareAuxiliariesSystem(int M, int N, Size Dc) : M(M), N(N), dimension_cut(Dc) {
         lazy::use_graph(graph);
         for (auto i = 0; i < M; i++) {
            auto& row = lattice.emplace_back();
            for (auto j = 0; j < N; j++) {
               row.push_back(lazy::Root<Tensor<T>>());
            }
         }

         up_to_down[-1] = lazy::Node([N]() { return std::vector<Tensor<T>>(N, Tensor<T>(1)); });
         for (auto i = 0; i < M; i++) {
            auto this_row = _collect_line(true, i);
            up_to_down[i] = lazy::Node(
                  [
#ifdef LAZY_DEBUG
                        i,
#endif
                        cut = dimension_cut](
                        const std::vector<Tensor<T>>& line_1, const std::vector<const Tensor<T>*>& line_2) -> std::vector<Tensor<T>> {
#ifdef LAZY_DEBUG
                     std::clog << "Calculating up to down two line to one line for row " << i << "\n";
#endif
                     auto result = _two_line_to_one_line("UDLR", line_1, line_2, cut);
#ifdef LAZY_DEBUG
                     for (const auto& i : result) {
                        using TAT::operator<<;
                        std::clog << i.names << "\n";
                     }
#endif
                     return result;
                  },
                  up_to_down[i - 1],
                  this_row);
         }

         down_to_up[M] = lazy::Node([N]() { return std::vector<Tensor<T>>(N, Tensor<T>(1)); });
         for (auto i = M; i-- > 0;) {
            auto this_row = _collect_line(true, i);
            down_to_up[i] = lazy::Node(
                  [
#ifdef LAZY_DEBUG
                        i,
#endif
                        cut = dimension_cut](
                        const std::vector<Tensor<T>>& line_1, const std::vector<const Tensor<T>*>& line_2) -> std::vector<Tensor<T>> {
#ifdef LAZY_DEBUG
                     std::clog << "Calculating down to up two line to one line for row " << i << "\n";
#endif
                     auto result = _two_line_to_one_line("DULR", line_1, line_2, cut);
#ifdef LAZY_DEBUG
                     for (const auto& i : result) {
                        using TAT::operator<<;
                        std::clog << i.names << "\n";
                     }
#endif
                     return result;
                  },
                  down_to_up[i + 1],
                  this_row);
         }

         left_to_right[-1] = lazy::Node([M]() { return std::vector<Tensor<T>>(M, Tensor<T>(1)); });
         for (auto j = 0; j < N; j++) {
            auto this_row = _collect_line(false, j);
            left_to_right[j] = lazy::Node(
                  [
#ifdef LAZY_DEBUG
                        j,
#endif
                        cut = dimension_cut](
                        const std::vector<Tensor<T>>& line_1, const std::vector<const Tensor<T>*>& line_2) -> std::vector<Tensor<T>> {
#ifdef LAZY_DEBUG
                     std::clog << "Calculating left to right two line to one line for column " << j << "\n";
#endif
                     auto result = _two_line_to_one_line("LRUD", line_1, line_2, cut);
#ifdef LAZY_DEBUG
                     for (const auto& i : result) {
                        using TAT::operator<<;
                        std::clog << i.names << "\n";
                     }
#endif
                     return result;
                  },
                  left_to_right[j - 1],
                  this_row);
         }

         right_to_left[N] = lazy::Node([M]() { return std::vector<Tensor<T>>(M, Tensor<T>(1)); });
         for (auto j = N; j-- > 0;) {
            auto this_row = _collect_line(false, j);
            right_to_left[j] = lazy::Node(
                  [
#ifdef LAZY_DEBUG
                        j,
#endif
                        cut = dimension_cut](
                        const std::vector<Tensor<T>>& line_1, const std::vector<const Tensor<T>*>& line_2) -> std::vector<Tensor<T>> {
#ifdef LAZY_DEBUG
                     std::clog << "Calculating right to left two line to one line for column " << j << "\n";
#endif
                     auto result = _two_line_to_one_line("RLUD", line_1, line_2, cut);
#ifdef LAZY_DEBUG
                     for (const auto& i : result) {
                        using TAT::operator<<;
                        std::clog << i.names << "\n";
                     }
#endif
                     return result;
                  },
                  right_to_left[j + 1],
                  this_row);
         }

         for (auto i = 0; i < M; i++) {
            for (auto j = -1; j < N; j++) {
               if (j == -1) {
                  left_to_right_3_1[i][j] = lazy::Node([]() { return Tensor<T>(1); });
                  left_to_right_3_3[i][j] = lazy::Node([]() { return Tensor<T>(1); });
               } else {
                  left_to_right_3_1[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              i,
#endif
                              j](const Tensor<T>& last_3_3, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating left to right 3 1 for " << i << j << "\n";
#endif
                           auto result = last_3_3.contract(last_line[j], {{"R1", "L"}}).edge_rename({{"R", "R1"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        left_to_right_3_3[i][j - 1],
                        up_to_down[i - 1]);
                  left_to_right_3_3[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              i,
#endif
                              j](const Tensor<T>& this_3_1, const Tensor<T>& this_site, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating left to right 3 3 for " << i << j << "\n";
#endif
                           auto result = this_3_1.contract(this_site, {{"R2", "L"}, {"D", "U"}})
                                               .edge_rename({{"R", "R2"}})
                                               .contract(last_line[j], {{"R3", "L"}, {"D", "U"}})
                                               .edge_rename({{"R", "R3"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        left_to_right_3_1[i][j],
                        lattice[i][j],
                        down_to_up[i + 1]);
               }
            }

            for (auto j = N + 1; j-- > 0;) {
               if (j == N) {
                  right_to_left_3_1[i][j] = lazy::Node([]() { return Tensor<T>(1); });
                  right_to_left_3_3[i][j] = lazy::Node([]() { return Tensor<T>(1); });
               } else {
                  right_to_left_3_1[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              i,
#endif
                              j](const Tensor<T>& last_3_3, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating right to left 3 1 for " << i << j << "\n";
#endif
                           auto result = last_3_3.contract(last_line[j], {{"L3", "R"}}).edge_rename({{"L", "L3"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        right_to_left_3_3[i][j + 1],
                        down_to_up[i + 1]);
                  right_to_left_3_3[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              i,
#endif
                              j](const Tensor<T>& this_3_1, const Tensor<T>& this_site, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating right to left 3 3 for " << i << j << "\n";
#endif
                           auto result = this_3_1.contract(this_site, {{"L2", "R"}, {"U", "D"}})
                                               .edge_rename({{"L", "L2"}})
                                               .contract(last_line[j], {{"L1", "R"}, {"U", "D"}})
                                               .edge_rename({{"L", "L1"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        right_to_left_3_1[i][j],
                        lattice[i][j],
                        up_to_down[i - 1]);
               }
            }
         }

         for (auto j = 0; j < N; j++) {
            for (auto i = -1; i < M; i++) {
               if (i == -1) {
                  up_to_down_3_1[i][j] = lazy::Node([]() { return Tensor<T>(1); });
                  up_to_down_3_3[i][j] = lazy::Node([]() { return Tensor<T>(1); });
               } else {
                  up_to_down_3_1[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              j,
#endif
                              i](const Tensor<T>& last_3_3, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating up to down 3 1 for " << i << j << "\n";
#endif
                           auto result = last_3_3.contract(last_line[i], {{"D1", "U"}}).edge_rename({{"D", "D1"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        up_to_down_3_3[i - 1][j],
                        left_to_right[j - 1]);
                  up_to_down_3_3[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              j,
#endif
                              i](const Tensor<T>& this_3_1, const Tensor<T>& this_site, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating up to down 3 3 for " << i << j << "\n";
#endif
                           auto result = this_3_1.contract(this_site, {{"D2", "U"}, {"R", "L"}})
                                               .edge_rename({{"D", "D2"}})
                                               .contract(last_line[i], {{"D3", "U"}, {"R", "L"}})
                                               .edge_rename({{"D", "D3"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        up_to_down_3_1[i][j],
                        lattice[i][j],
                        right_to_left[j + 1]);
               }
            }

            for (auto i = M + 1; i-- > 0;) {
               if (i == M) {
                  down_to_up_3_1[i][j] = lazy::Node([]() { return Tensor<T>(1); });
                  down_to_up_3_3[i][j] = lazy::Node([]() { return Tensor<T>(1); });
               } else {
                  down_to_up_3_1[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              j,
#endif
                              i](const Tensor<T>& last_3_3, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating down to up 3 1 for " << i << j << "\n";
#endif
                           auto result = last_3_3.contract(last_line[i], {{"U3", "D"}}).edge_rename({{"U", "U3"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        down_to_up_3_3[i + 1][j],
                        right_to_left[j + 1]);
                  down_to_up_3_3[i][j] = lazy::Node(
                        [
#ifdef LAZY_DEBUG
                              j,
#endif
                              i](const Tensor<T>& this_3_1, const Tensor<T>& this_site, const std::vector<Tensor<T>>& last_line) {
#ifdef LAZY_DEBUG
                           std::clog << "Calculating down to up 3 3 for " << i << j << "\n";
#endif
                           auto result = this_3_1.contract(this_site, {{"U2", "D"}, {"L", "R"}})
                                               .edge_rename({{"U", "U2"}})
                                               .contract(last_line[i], {{"U1", "D"}, {"L", "R"}})
                                               .edge_rename({{"U", "U1"}});
#ifdef LAZY_DEBUG

                           using TAT::operator<<;
                           std::clog << result.names << "\n";

#endif
                           return result;
                        },
                        down_to_up_3_1[i][j],
                        lattice[i][j],
                        left_to_right[j - 1]);
               }
            }
         }

         lazy::use_graph();
      }

      // TODO hint and hole
      auto operator()(const std::map<std::tuple<int, int>, Tensor<T>>& replacement) const {
         if (replacement.size() == 0) {
            return left_to_right_3_3.at(M - 1).at(N - 1)->get();
         } else if (replacement.size() == 1) {
            auto [i, j] = replacement.begin()->first;
            const auto& new_tensor = replacement.begin()->second;
            return (left_to_right_3_1.at(i).at(j)->get())
                  .contract(new_tensor.edge_rename({{"R", "R2"}}), {{"R2", "L"}, {"D", "U"}})
                  .contract(right_to_left_3_1.at(i).at(j)->get(), {{"R1", "L1"}, {"R2", "L2"}, {"R3", "L3"}, {"D", "U"}});
         } else if (replacement.size() == 2) {
            auto iter = replacement.begin();
            auto [x1, y1] = iter->first;
            const auto& new_tensor_1 = iter->second;
            ++iter;
            auto [x2, y2] = iter->first;
            const auto& new_tensor_2 = iter->second;
            if (x1 == x2) {
               if (y1 + 1 == y2) {
                  return (left_to_right_3_1.at(x1).at(y1)->get())
                        .contract(new_tensor_1.edge_rename({{"R", "R2"}}), {{"R2", "L"}, {"D", "U"}})
                        .contract(down_to_up.at(x1 + 1)->get()[y1].edge_rename({{"R", "R3"}}), {{"R3", "L"}, {"D", "U"}})
                        .contract(up_to_down.at(x2 - 1)->get()[y2].edge_rename({{"R", "R1"}}), {{"R1", "L"}})
                        .contract(new_tensor_2.edge_rename({{"R", "R2"}}), {{"R2", "L"}, {"D", "U"}})
                        .contract(right_to_left_3_1.at(x2).at(y2)->get(), {{"R1", "L1"}, {"R2", "L2"}, {"R3", "L3"}, {"D", "U"}});
               }
               if (y2 + 1 == y1) {
                  return (left_to_right_3_1.at(x2).at(y2)->get())
                        .contract(new_tensor_2.edge_rename({{"R", "R2"}}), {{"R2", "L"}, {"D", "U"}})
                        .contract(down_to_up.at(x2 + 1)->get()[y2].edge_rename({{"R", "R3"}}), {{"R3", "L"}, {"D", "U"}})
                        .contract(up_to_down.at(x1 - 1)->get()[y1].edge_rename({{"R", "R1"}}), {{"R1", "L"}})
                        .contract(new_tensor_1.edge_rename({{"R", "R2"}}), {{"R2", "L"}, {"D", "U"}})
                        .contract(right_to_left_3_1.at(x1).at(y1)->get(), {{"R1", "L1"}, {"R2", "L2"}, {"R3", "L3"}, {"D", "U"}});
               }
            }
            if (y1 == y2) {
               if (x1 + 1 == x2) {
                  return (up_to_down_3_1.at(x1).at(y1)->get())
                        .contract(new_tensor_1.edge_rename({{"D", "D2"}}), {{"D2", "U"}, {"R", "L"}})
                        .contract(right_to_left.at(y1 + 1)->get()[x1].edge_rename({{"D", "D3"}}), {{"D3", "U"}, {"R", "L"}})
                        .contract(left_to_right.at(y2 - 1)->get()[x2].edge_rename({{"D", "D1"}}), {{"D1", "U"}})
                        .contract(new_tensor_2.edge_rename({{"D", "D2"}}), {{"D2", "U"}, {"R", "L"}})
                        .contract(down_to_up_3_1.at(x2).at(y2)->get(), {{"D1", "U1"}, {"D2", "U2"}, {"D3", "U3"}, {"R", "L"}});
               }
               if (x2 + 1 == x1) {
                  return (up_to_down_3_1.at(x2).at(y2)->get())
                        .contract(new_tensor_2.edge_rename({{"D", "D2"}}), {{"D2", "U"}, {"R", "L"}})
                        .contract(right_to_left.at(y2 + 1)->get()[x1].edge_rename({{"D", "D3"}}), {{"D3", "U"}, {"R", "L"}})
                        .contract(left_to_right.at(y1 - 1)->get()[x2].edge_rename({{"D", "D1"}}), {{"D1", "U"}})
                        .contract(new_tensor_1.edge_rename({{"D", "D2"}}), {{"D2", "U"}, {"R", "L"}})
                        .contract(down_to_up_3_1.at(x1).at(y1)->get(), {{"D1", "U1"}, {"D2", "U2"}, {"D3", "U3"}, {"R", "L"}});
               }
            }
         }
         throw NotImplementedError("Unsupported replacement style");
      }

      static std::vector<Tensor<T>>
      _two_line_to_one_line(const char* udlr_name, const std::vector<Tensor<T>>& line_1, const std::vector<const Tensor<T>*>& line_2, Size cut) {
         std::string up = {udlr_name[0]};
         std::string down = {udlr_name[1]};
         std::string left = {udlr_name[2]};
         std::string right = {udlr_name[3]};
         char suffix_1 = '1';
         char suffix_2 = '2';
         auto up1 = up + suffix_1;
         auto up2 = up + suffix_2;
         auto down1 = down + suffix_1;
         auto down2 = down + suffix_2;
         auto left1 = left + suffix_1;
         auto left2 = left + suffix_2;
         auto right1 = right + suffix_1;
         auto right2 = right + suffix_2;

         auto length = line_1.size();
         if (line_2.size() != length) {
            throw std::logic_error("Different Length in Two Line to One Line");
         }
         auto double_line = std::vector<Tensor<T>>();
         for (auto i = 0; i < length; i++) {
            double_line.push_back(line_1[i]
                                        .edge_rename({{left, left1}, {right, right1}})
                                        .contract(line_2[i]->edge_rename({{left, left2}, {right, right2}}), {{down, up}}));
         }
         for (auto i = 0; i < length - 1; i++) {
            // 虽然实际上是range(length - 2), 但是多计算一个以免角标merge的麻烦
            auto [q, r] = double_line[i].qr('r', {right1, right2}, right, left);
            double_line[i] = std::move(q);
            double_line[i + 1] = double_line[i + 1].contract(r, {{left1, right1}, {left2, right2}});
         }
         for (auto i = length - 1; i-- > 0;) {
            //可能上下都有
            auto [u, s, v] = double_line[i]
                                   .edge_rename({{up, up1}, {down, down1}})
                                   .contract(double_line[i + 1].edge_rename({{up, up2}, {down, down2}}), {{right, left}})
                                   .svd({left, up1, down1}, right, left, cut);
            double_line[i + 1] = v.edge_rename({{up2, up}, {down2, down}});
            double_line[i] = u.multiple(s, right, 'u').edge_rename({{up1, up}, {down1, down}});
         }
         return double_line;
      }

      auto _collect_line(bool horizontal, int number) const {
         if (horizontal) {
            auto this_line = lazy::Path([]() -> std::vector<const Tensor<T>*> { return std::vector<const Tensor<T>*>(); });
            for (auto j = 0; j < N; j++) {
               this_line = lazy::Path(
                     [](std::vector<const Tensor<T>*> this_line_origin, const Tensor<T>& another_tensor) -> std::vector<const Tensor<T>*> {
                        this_line_origin.push_back(&another_tensor);
                        return this_line_origin;
                     },
                     this_line,
                     lattice[number][j]);
            }
            return this_line;
         } else {
            auto this_line = lazy::Path([]() -> std::vector<const Tensor<T>*> { return std::vector<const Tensor<T>*>(); });
            for (auto i = 0; i < M; i++) {
               this_line = lazy::Path(
                     [](std::vector<const Tensor<T>*> this_line_origin, const Tensor<T>& another_tensor) -> std::vector<const Tensor<T>*> {
                        this_line_origin.push_back(&another_tensor);
                        return this_line_origin;
                     },
                     this_line,
                     lattice[i][number]);
            }
            return this_line;
         }
      }
   };
}; // namespace square
#endif