/**
 * \file simple_update_lattice.hpp
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
#ifndef SQUARE_SIMPLE_UPDATE_LATTICE_HPP
#define SQUARE_SIMPLE_UPDATE_LATTICE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "abstract_network_lattice.hpp"

namespace square {
   template<typename T>
   struct SimpleUpdateLattice : AbstractNetworkLattice<T> {
      std::map<std::tuple<char, int, int>, typename Tensor<T>::SingularType> environment;

      using AbstractNetworkLattice<T>::dimension_virtual;
      using AbstractNetworkLattice<T>::lattice;
      using AbstractLattice<T>::M;
      using AbstractLattice<T>::N;
      using AbstractLattice<T>::dimension_physics;
      using AbstractLattice<T>::hamiltonians;

      SimpleUpdateLattice() : AbstractNetworkLattice<T>(), environment(){};
      SimpleUpdateLattice(const SimpleUpdateLattice<T>&) = default;
      SimpleUpdateLattice(SimpleUpdateLattice<T>&&) = default;
      SimpleUpdateLattice<T>& operator=(const SimpleUpdateLattice<T>&) = default;
      SimpleUpdateLattice<T>& operator=(SimpleUpdateLattice<T>&&) = default;

      SimpleUpdateLattice(int M, int N, Size D, Size d) : AbstractNetworkLattice<T>(M, N, D, d) {}

      explicit SimpleUpdateLattice(const SamplingGradientLattice<T>& other);

      void update(int total_step, real<T> delta_t, Size new_dimension) {
         if (new_dimension) {
            dimension_virtual = new_dimension;
         }
         TAT::mpi.out() << clear_line << "Simple updating start, total_step=" << total_step << ", dimension=" << dimension_virtual
                        << ", delta_t=" << delta_t << "\n"
                        << std::flush;
         std::map<const Tensor<T>*, std::shared_ptr<const Tensor<T>>> updater_pool;
         std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>> updater;
         for (const auto& [positions, term] : hamiltonians) {
            auto position_number = positions.size();
            auto term_pointer = &*term;
            if (auto found = updater_pool.find(term_pointer); found == updater_pool.end()) {
               auto pairs = std::set<std::tuple<Name, Name>>();
               for (auto i = 0; i < position_number; i++) {
                  pairs.insert({"I" + std::to_string(i), "O" + std::to_string(i)});
               }
               updater[positions] = updater_pool[term_pointer] = std::make_shared<const Tensor<T>>((-delta_t * *term).exponential(pairs, 8));
            } else {
               updater[positions] = found->second;
            }
         }
         auto positions_sequence = _simple_update_positions_sequence();
         for (auto step = 0; step < total_step; step++) {
            for (auto iter = positions_sequence.begin(); iter != positions_sequence.end(); ++iter) {
               _single_group_simple_update(*iter, updater);
            }
            for (auto iter = positions_sequence.rbegin(); iter != positions_sequence.rend(); ++iter) {
               _single_group_simple_update(*iter, updater);
            }
            TAT::mpi.out() << clear_line << "Simple updating, total_step=" << total_step << ", dimension=" << dimension_virtual
                           << ", delta_t=" << delta_t << ", step=" << (step + 1) << "\r" << std::flush;
         }
         TAT::mpi.out() << clear_line << "Simple update done, total_step=" << total_step << ", dimension=" << dimension_virtual
                        << ", delta_t=" << delta_t << "\n"
                        << std::flush;
      }

      void _single_group_simple_update(
            const std::vector<std::vector<std::tuple<int, int>>>& group,
            const std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>>& updater) {
         for (auto i = 0; i < group.size(); i++) {
            if (i % TAT::mpi.size == TAT::mpi.rank) {
               const auto& positions = group[i];
               const auto& tensor = *updater.at(positions);
               _single_term_simple_update(positions, tensor);
            }
         }
         MPI_Barrier(MPI_COMM_WORLD);
         for (auto i = 0; i < group.size(); i++) {
            const auto& positions = group[i];
            auto root = i % TAT::mpi.size;
            for (const auto& [x, y] : positions) {
               lattice[x][y] = lattice[x][y].broadcast(root);
            }
            for (const auto& [d, x, y] : _get_related_environment(positions)) {
               environment[{d, x, y}] = TAT::mpi.broadcast(environment[{d, x, y}], root);
            }
         }
      }

      static std::vector<std::tuple<char, int, int>> _get_related_environment(const std::vector<std::tuple<int, int>>& positions) {
         if (positions.size() == 1) {
            return {};
         }
         if (positions.size() == 2) {
            const auto& [x1, y1] = positions[0];
            const auto& [x2, y2] = positions[1];
            if (x1 == x2 && y1 + 1 == y2) {
               return {{'R', x1, y1}};
            }
            if (x1 == x2 && y1 == y2 + 1) {
               return {{'R', x2, y2}};
            }
            if (x1 + 1 == x2 && y1 == y2) {
               return {{'D', x1, y1}};
            }
            if (x1 == x2 + 1 && y1 == y2) {
               return {{'D', x2, y2}};
            }
         }
         throw NotImplementedError("Unsupported environment style");
      }

      auto _simple_update_positions_sequence() const {
         // 以后simple update如果要并行，下面之中每类内都是无依赖的
         auto result = std::vector<std::vector<std::vector<std::tuple<int, int>>>>();
         // 应该不存在常数项的hamiltonians
         // 单点
         auto& group0 = result.emplace_back();
         for (const auto& [positions, tensor] : hamiltonians) {
            if (positions.size() == 1) {
               group0.push_back(positions);
            }
         }
         // 双点，分成四大类+其他
         auto& group1 = result.emplace_back();
         for (const auto& [positions, tensor] : hamiltonians) {
            if (positions.size() == 2) {
               const auto& [x1, y1] = positions[0];
               const auto& [x2, y2] = positions[1];
               if (x1 == x2 && ((y1 + 1 == y2 && y1 % 2 == 0) || (y1 == y2 + 1 && y2 % 2 == 0))) {
                  group1.push_back(positions);
               }
            }
         }
         auto& group2 = result.emplace_back();
         for (const auto& [positions, tensor] : hamiltonians) {
            if (positions.size() == 2) {
               const auto& [x1, y1] = positions[0];
               const auto& [x2, y2] = positions[1];
               if (x1 == x2 && ((y1 + 1 == y2 && y1 % 2 == 1) || (y1 == y2 + 1 && y2 % 2 == 1))) {
                  group2.push_back(positions);
               }
            }
         }
         auto& group3 = result.emplace_back();
         for (const auto& [positions, tensor] : hamiltonians) {
            if (positions.size() == 2) {
               const auto& [x1, y1] = positions[0];
               const auto& [x2, y2] = positions[1];
               if (y1 == y2 && ((x1 + 1 == x2 && x1 % 2 == 0) || (x1 == x2 + 1 && x2 % 2 == 0))) {
                  group3.push_back(positions);
               }
            }
         }
         auto& group4 = result.emplace_back();
         for (const auto& [positions, tensor] : hamiltonians) {
            if (positions.size() == 2) {
               const auto& [x1, y1] = positions[0];
               const auto& [x2, y2] = positions[1];
               if (y1 == y2 && ((x1 + 1 == x2 && x1 % 2 == 1) || (x1 == x2 + 1 && x2 % 2 == 1))) {
                  group4.push_back(positions);
               }
            }
         }
         // 其他类型和更多的格点暂时不支持
         auto total_size = 0;
         for (const auto& group : result) {
            total_size += group.size();
         }
         if (total_size != hamiltonians.size()) {
            throw NotImplementedError("Unsupported simple update style");
         }
         return result;
      }

      void _single_term_simple_update(const std::vector<std::tuple<int, int>>& positions, const Tensor<T>& updater) {
         if (positions.size() == 0) {
         } else if (positions.size() == 1) {
            _single_term_simple_update_single_site(positions[0], updater);
            return;
         } else if (positions.size() == 2) {
            const auto& position_1 = positions[0];
            const auto& position_2 = positions[1];
            if (std::get<0>(position_1) == std::get<0>(position_2)) {
               if (std::get<1>(position_1) == std::get<1>(position_2) + 1) {
                  _single_term_simple_update_horizontal_bond(position_2, updater);
                  return;
               }
               if (std::get<1>(position_1) + 1 == std::get<1>(position_2)) {
                  _single_term_simple_update_horizontal_bond(position_1, updater);
                  return;
               }
            }
            if (std::get<1>(position_1) == std::get<1>(position_2)) {
               if (std::get<0>(position_1) == std::get<0>(position_2) + 1) {
                  _single_term_simple_update_vertical_bond(position_2, updater);
                  return;
               }
               if (std::get<0>(position_1) + 1 == std::get<0>(position_2)) {
                  _single_term_simple_update_vertical_bond(position_1, updater);
                  return;
               }
            }
         }
         throw NotImplementedError("Unsupported simple update style");
      }

      void _single_term_simple_update_single_site(const std::tuple<int, int>& position, const Tensor<T>& updater) {
         auto [i, j] = position;
         lattice[i][j] = lattice[i][j].contract(updater, {{"P", "I0"}}).edge_rename({{"O0", "P"}});
      }

      void _single_term_simple_update_horizontal_bond(const std::tuple<int, int>& position, const Tensor<T>& updater) {
         auto [i, j] = position;
         auto left = lattice[i][j];
         left = try_multiple(left, i, j, 'L');
         left = try_multiple(left, i, j, 'U');
         left = try_multiple(left, i, j, 'D');
         left = try_multiple(left, i, j, 'R');
         auto right = lattice[i][j + 1];
         right = try_multiple(right, i, j + 1, 'U');
         right = try_multiple(right, i, j + 1, 'D');
         right = try_multiple(right, i, j + 1, 'R');
         auto [left_q, left_r] = left.qr('r', {"P", "R"}, "R", "L");
         auto [right_q, right_r] = right.qr('r', {"P", "L"}, "L", "R");
         auto [u, s, v] = left_r.edge_rename({{"P", "P0"}})
                                .contract(right_r.edge_rename({{"P", "P1"}}), {{"R", "L"}})
                                .contract(updater, {{"P0", "I0"}, {"P1", "I1"}})
                                .svd({"L", "O0"}, "R", "L", dimension_virtual);
         s /= s.template norm<-1>();
         environment[{'R', i, j}] = std::move(s);
         u = u.contract(left_q, {{"L", "R"}}).edge_rename({{"O0", "P"}});
         u = try_multiple(u, i, j, 'L', true);
         u = try_multiple(u, i, j, 'U', true);
         u = try_multiple(u, i, j, 'D', true);
         u /= u.template norm<-1>();
         lattice[i][j] = std::move(u);
         v = v.contract(right_q, {{"R", "L"}}).edge_rename({{"O1", "P"}});
         v = try_multiple(v, i, j + 1, 'U', true);
         v = try_multiple(v, i, j + 1, 'D', true);
         v = try_multiple(v, i, j + 1, 'R', true);
         v /= v.template norm<-1>();
         lattice[i][j + 1] = std::move(v);
      }

      void _single_term_simple_update_vertical_bond(const std::tuple<int, int>& position, const Tensor<T>& updater) {
         auto [i, j] = position;
         auto up = lattice[i][j];
         up = try_multiple(up, i, j, 'L');
         up = try_multiple(up, i, j, 'U');
         up = try_multiple(up, i, j, 'D');
         up = try_multiple(up, i, j, 'R');
         auto down = lattice[i + 1][j];
         down = try_multiple(down, i + 1, j, 'L');
         down = try_multiple(down, i + 1, j, 'D');
         down = try_multiple(down, i + 1, j, 'R');
         auto [up_q, up_r] = up.qr('r', {"P", "D"}, "D", "U");
         auto [down_q, down_r] = down.qr('r', {"P", "U"}, "U", "D");
         auto [u, s, v] = up_r.edge_rename({{"P", "P0"}})
                                .contract(down_r.edge_rename({{"P", "P1"}}), {{"D", "U"}})
                                .contract(updater, {{"P0", "I0"}, {"P1", "I1"}})
                                .svd({"U", "O0"}, "D", "U", dimension_virtual);
         s /= s.template norm<-1>();
         environment[{'D', i, j}] = std::move(s);
         u = u.contract(up_q, {{"U", "D"}}).edge_rename({{"O0", "P"}});
         u = try_multiple(u, i, j, 'L', true);
         u = try_multiple(u, i, j, 'U', true);
         u = try_multiple(u, i, j, 'R', true);
         u /= u.template norm<-1>();
         lattice[i][j] = std::move(u);
         v = v.contract(down_q, {{"D", "U"}}).edge_rename({{"O1", "P"}});
         v = try_multiple(v, i + 1, j, 'L', true);
         v = try_multiple(v, i + 1, j, 'D', true);
         v = try_multiple(v, i + 1, j, 'R', true);
         v /= v.template norm<-1>();
         lattice[i + 1][j] = std::move(v);
      }

      auto try_multiple(Tensor<T> tensor, int i, int j, char direction, bool division = false) const {
         if (direction == 'L') {
            if (auto found = environment.find(std::tuple<char, int, int>('R', i, j - 1)); found != environment.end()) {
               return tensor.multiple(found->second, "L", 'v', division);
            }
         }
         if (direction == 'U') {
            if (auto found = environment.find(std::tuple<char, int, int>('D', i - 1, j)); found != environment.end()) {
               return tensor.multiple(found->second, "U", 'v', division);
            }
         }
         if (direction == 'D') {
            if (auto found = environment.find(std::tuple<char, int, int>('D', i, j)); found != environment.end()) {
               return tensor.multiple(found->second, "D", 'u', division);
            }
         }
         if (direction == 'R') {
            if (auto found = environment.find(std::tuple<char, int, int>('R', i, j)); found != environment.end()) {
               return tensor.multiple(found->second, "R", 'u', division);
            }
         }
         return tensor;
      }
   };

   template<typename T>
   std::ostream& operator<(std::ostream& out, const SimpleUpdateLattice<T>& lattice) {
      using TAT::operator<;
      out < static_cast<const AbstractNetworkLattice<T>&>(lattice);
      Size map_size = lattice.environment.size();
      out < map_size;
      for (const auto& [position, singular] : lattice.environment) {
         out < position < singular;
      }
      return out;
   }

   template<typename T>
   std::istream& operator>(std::istream& in, SimpleUpdateLattice<T>& lattice) {
      using TAT::operator>;
      in > static_cast<AbstractNetworkLattice<T>&>(lattice);
      Size map_size;
      in > map_size;
      lattice.environment.clear();
      for (auto i = 0; i < map_size; i++) {
         std::tuple<char, int, int> position;
         typename Tensor<T>::SingularType singular;
         in > position > singular;
         lattice.environment[std::move(position)] = std::move(singular);
      }
      return in;
   }
} // namespace square

#endif
