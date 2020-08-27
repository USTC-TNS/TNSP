/**
 * \file tools.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_TOOLS_HPP
#define TAT_TOOLS_HPP

#include "tensor.hpp"

namespace TAT {
   namespace tools {
      // default: up to down
      template<class ScalarType, class Symmetry>
      auto two_line_to_one_line(
            const std::array<Name, 4>& udlr_name,
            std::vector<const Tensor<ScalarType, Symmetry>*> line_1,
            std::vector<const Tensor<ScalarType, Symmetry>*> line_2,
            Size cut) {
         const auto& [up, down, left, right] = udlr_name;
         const Name up1 = id_to_name.at(up.id) + "_1";
         const Name up2 = id_to_name.at(up.id) + "_2";
         const Name down1 = id_to_name.at(down.id) + "_1";
         const Name down2 = id_to_name.at(down.id) + "_2";
         const Name left1 = id_to_name.at(left.id) + "_1";
         const Name left2 = id_to_name.at(left.id) + "_2";
         const Name right1 = id_to_name.at(right.id) + "_1";
         const Name right2 = id_to_name.at(right.id) + "_2";

         int length = line_1.size();
         if (length != line_2.size()) {
            warning_or_error("Different Length When Do Two Line to One Line");
         }
         //std::clog << "Two Line to One Line Start\n";

         // product
         std::vector<Tensor<ScalarType, Symmetry>> double_line;
         //std::clog << "double line:\n";
         for (int i = 0; i < length; i++) {
            double_line.push_back(Tensor<ScalarType, Symmetry>::contract(
                  line_1[i]->edge_rename({{left, left1}, {right, right1}}), line_2[i]->edge_rename({{left, left2}, {right, right2}}), {{down, up}}));
            //std::clog << double_line[i] << "\n";
         }

         // left canonicalize
         for (int i = 0; i < length - 1; i++) {
            // lattice: 0 ~ L-1
            // 0 ... L-2
            auto [u, s, v] = double_line[i].svd({right1, right2}, left, right);
            double_line[i] = std::move(v);
            double_line[i + 1] = double_line[i + 1].contract(u, {{left1, right1}, {left2, right2}}).multiple(s, left, false);
         }
         //std::clog << "double line:\n";
         //for (int i = 0; i < length; i++) {
         //   std::clog << double_line[i] << "\n";
         //}

         // right svd
         for (int i = length - 2; i >= 0; i--) {
            // L-2 and L-1, ... 0 and 1
            // i and i+1
            // get name of double_line[i]
            auto u_names = std::set<Name>(double_line[i].names.begin(), double_line[i].names.end());
            u_names.erase(right);
            u_names.erase(up);
            u_names.erase(down);
            u_names.insert(up1);
            u_names.insert(down1);
            auto [u, s, v] = Tensor<ScalarType, Symmetry>::contract(
                                   double_line[i].edge_rename({{up, up1}, {down, down1}}),
                                   double_line[i + 1].edge_rename({{up, up2}, {down, down2}}),
                                   {{right, left}})
                                   .svd(u_names, right, left, cut);
            double_line[i + 1] = v.edge_rename({{up2, up}, {down2, down}});
            double_line[i] = u.multiple(s, right, false).edge_rename({{up1, up}, {down1, down}});
         }

         //std::clog << "Two Line to One Line End\n";
         return double_line;
      }

      template<class ScalarType, class Symmetry, class Key>
      struct network {
         std::map<Key, Tensor<ScalarType, Symmetry>> site;
         // Singular is map<Symmetry, vector>, it there is no environment, the map is empty
         std::map<std::tuple<Key, Key>, std::map<std::tuple<Name, Name>, typename Tensor<ScalarType, Symmetry>::Singular>> bond;
         // TODO: network
      };
   } // namespace tools
} // namespace TAT
#endif
