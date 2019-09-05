/**
 * \file example/PEPS_GO.dir/use_Auxiliary.hpp
 *
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef TAT_USE_AUXILIARY_HPP_
#define TAT_USE_AUXILIARY_HPP_

#include "Configuration.hpp"

void Configuration::calculate_hole() {
      // up and down
      for (int i = 0; i < peps.metadata.M; i++) {
            // left
            auto left1 = std::vector<Node>(peps.metadata.N);
            auto left2 = std::vector<Node>(peps.metadata.N);
            auto left3 = std::vector<Node>(peps.metadata.N);
            auto leg_right1 = TAT::Legs("calc_hole_right1");
            auto leg_right2 = TAT::Legs("calc_hole_right2");
            auto leg_right3 = TAT::Legs("calc_hole_right3");
            {
                  for (int j = 0; j < peps.metadata.N; j++) {
                        if (i == 0 && j == 0) {
                              // left1 not exist
                        } else if (i == 0) {
                              left1[j] = left3[j - 1];
                        } else if (j == 0) {
                              left1[j] = aux.up_to_down[{i - 1, j}].legs_rename({{TAT::legs_name::Right, leg_right1}});
                        } else {
                              left1[j] = Node::contract(
                                    left3[j - 1],
                                    aux.up_to_down[{i - 1, j}].legs_rename({{TAT::legs_name::Right, leg_right1}}),
                                    {leg_right1},
                                    {TAT::legs_name::Left});
                        }

                        if (i == 0 && j == 0) {
                              // left1 not exist
                              left2[j] = aux.down_to_up[{i + 1, j}].legs_rename({{TAT::legs_name::Right, leg_right2}});
                        } else if (i == peps.metadata.M - 1) {
                              left2[j] = left1[j];
                        } else {
                              left2[j] = Node::contract(
                                    left1[j],
                                    aux.down_to_up[{i + 1, j}].legs_rename({{TAT::legs_name::Right, leg_right2}}),
                                    {leg_right2},
                                    {TAT::legs_name::Left});
                        }

                        left3[j] = Node::contract(
                              left2[j],
                              lattice[{i, j}].legs_rename({{TAT::legs_name::Right, leg_right3}}),
                              {leg_right3, TAT::legs_name::Down, TAT::legs_name::Up},
                              {TAT::legs_name::Left, TAT::legs_name::Up, TAT::legs_name::Down});
                  }
            }
            // right
            auto right1 = std::vector<Node>(peps.metadata.N);
            auto right2 = std::vector<Node>(peps.metadata.N);
            auto right3 = std::vector<Node>(peps.metadata.N);
            auto leg_left1 = TAT::Legs("calc_hole_left1");
            auto leg_left2 = TAT::Legs("calc_hole_left2");
            auto leg_left3 = TAT::Legs("calc_hole_left3");
            {
                  for (int j = peps.metadata.N - 1; j >= 0; j--) {
                        if (i == 0 && j == peps.metadata.N - 1) {
                              // right1 not exist
                        } else if (i == 0) {
                              right1[j] = right3[j + 1];
                        } else if (j == peps.metadata.N - 1) {
                              right1[j] = aux.up_to_down[{i - 1, j}].legs_rename({{TAT::legs_name::Left, leg_left1}});
                        } else {
                              right1[j] = Node::contract(
                                    right3[j + 1],
                                    aux.up_to_down[{i - 1, j}].legs_rename({{TAT::legs_name::Left, leg_left1}}),
                                    {leg_left1},
                                    {TAT::legs_name::Right});
                        }

                        if (i == 0 && j == peps.metadata.N - 1) {
                              // right1 not exist
                              right2[j] = aux.down_to_up[{i + 1, j}].legs_rename({{TAT::legs_name::Left, leg_left2}});
                        } else if (i == peps.metadata.M - 1) {
                              right2[j] = right1[j];
                        } else {
                              right2[j] = Node::contract(
                                    right1[j],
                                    aux.down_to_up[{i + 1, j}].legs_rename({{TAT::legs_name::Left, leg_left2}}),
                                    {leg_left2},
                                    {TAT::legs_name::Right});
                        }

                        right3[j] = Node::contract(
                              right2[j],
                              lattice[{i, j}].legs_rename({{TAT::legs_name::Left, leg_left3}}),
                              {leg_left3, TAT::legs_name::Down, TAT::legs_name::Up},
                              {TAT::legs_name::Right, TAT::legs_name::Up, TAT::legs_name::Down});
                  }
            }
            // ws and grad single hole and part of double hole
            for (int j = 0; j < peps.metadata.N; j++) {
                  if (j == peps.metadata.N - 1) {
                        target.gradient[{i, j}] = left2[j].legs_rename({{leg_right3, TAT::legs_name::Left},
                                                                        {TAT::legs_name::Down, TAT::legs_name::Up},
                                                                        {TAT::legs_name::Up, TAT::legs_name::Down}});
                  } else {
                        target.gradient[{i, j}] = Node::contract(
                              left2[j].legs_rename({{leg_right3, TAT::legs_name::Left},
                                                    {TAT::legs_name::Down, TAT::legs_name::Up},
                                                    {TAT::legs_name::Up, TAT::legs_name::Down}}),
                              right3[j + 1].legs_rename({{leg_left3, TAT::legs_name::Right}}),
                              {leg_right1, leg_right2},
                              {leg_left1, leg_left2});
                  }
                  for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                        target.single_hole[{i, j, k}] = target.gradient[{i, j}].partial_contract(
                              peps.lattice[{i, j, k}],
                              {TAT::legs_name::Up, TAT::legs_name::Down, TAT::legs_name::Left, TAT::legs_name::Right},
                              {TAT::legs_name::Up, TAT::legs_name::Down, TAT::legs_name::Left, TAT::legs_name::Right});
                  }
                  if (j != peps.metadata.N - 1) {
                        for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                              auto tmp_left = left2[j].partial_contract(
                                    peps.lattice[{i, j, k}],
                                    {TAT::legs_name::Up, TAT::legs_name::Down, leg_right3},
                                    {TAT::legs_name::Down, TAT::legs_name::Up, TAT::legs_name::Left});
                              for (TAT::Size l = 0; l < peps.metadata.d; l++) {
                                    auto tmp_right = right2[j + 1].partial_contract(
                                          peps.lattice[{i, j + 1, l}],
                                          {TAT::legs_name::Up, TAT::legs_name::Down, leg_left3},
                                          {TAT::legs_name::Down, TAT::legs_name::Up, TAT::legs_name::Right});
                                    target.double_hole[{i, j, TAT::legs_name::Right, k, l}] = Node::contract(
                                          tmp_left,
                                          tmp_right,
                                          {leg_right1, leg_right2, TAT::legs_name::Right},
                                          {leg_left1, leg_left2, TAT::legs_name::Left});
                                    target.double_hole[{i, j + 1, TAT::legs_name::Left, l, k}] =
                                          target.double_hole[{i, j, TAT::legs_name::Right, k, l}];
                              }
                        }
                  }
            }
      }
      // left and right
      for (int j = 0; j < peps.metadata.N; j++) {
            // up
            auto up1 = std::vector<Node>(peps.metadata.M);
            auto up2 = std::vector<Node>(peps.metadata.M);
            auto up3 = std::vector<Node>(peps.metadata.M);
            auto leg_down1 = TAT::Legs("calc_hole_down1");
            auto leg_down2 = TAT::Legs("calc_hole_down2");
            auto leg_down3 = TAT::Legs("calc_hole_down3");
            {
                  for (int i = 0; i < peps.metadata.M; i++) {
                        if (i == 0 && j == 0) {
                              // up1 not exist
                        } else if (j == 0) {
                              up1[i] = up3[i - 1];
                        } else if (i == 0) {
                              up1[i] = aux.left_to_right[{i, j - 1}].legs_rename({{TAT::legs_name::Down, leg_down1}});
                        } else {
                              up1[i] = Node::contract(
                                    up3[i - 1],
                                    aux.left_to_right[{i, j - 1}].legs_rename({{TAT::legs_name::Down, leg_down1}}),
                                    {leg_down1},
                                    {TAT::legs_name::Up});
                        }

                        if (i == 0 && j == 0) {
                              // up1 not exist
                              up2[i] = aux.right_to_left[{i, j + 1}].legs_rename({{TAT::legs_name::Down, leg_down2}});
                        } else if (j == peps.metadata.N - 1) {
                              up2[i] = up1[i];
                        } else {
                              up2[i] = Node::contract(
                                    up1[i],
                                    aux.right_to_left[{i, j + 1}].legs_rename({{TAT::legs_name::Down, leg_down2}}),
                                    {leg_down2},
                                    {TAT::legs_name::Up});
                        }

                        up3[i] = Node::contract(
                              up2[i],
                              lattice[{i, j}].legs_rename({{TAT::legs_name::Down, leg_down3}}),
                              {leg_down3, TAT::legs_name::Left, TAT::legs_name::Right},
                              {TAT::legs_name::Up, TAT::legs_name::Right, TAT::legs_name::Left});
                  }
            }
            auto down1 = std::vector<Node>(peps.metadata.M);
            auto down2 = std::vector<Node>(peps.metadata.M);
            auto down3 = std::vector<Node>(peps.metadata.M);
            auto leg_up1 = TAT::Legs("calc_hole_up1");
            auto leg_up2 = TAT::Legs("calc_hole_up2");
            auto leg_up3 = TAT::Legs("calc_hole_up3");
            {
                  for (int i = peps.metadata.M - 1; i >= 0; i--) {
                        if (i == peps.metadata.M - 1 && j == 0) {
                              // down1 not exist
                        } else if (j == 0) {
                              down1[i] = down3[i + 1];
                        } else if (i == peps.metadata.M - 1) {
                              down1[i] = aux.left_to_right[{i, j - 1}].legs_rename({{TAT::legs_name::Up, leg_up1}});
                        } else {
                              down1[i] = Node::contract(
                                    down3[i + 1],
                                    aux.left_to_right[{i, j - 1}].legs_rename({{TAT::legs_name::Up, leg_up1}}),
                                    {leg_up1},
                                    {TAT::legs_name::Down});
                        }

                        if (i == peps.metadata.M - 1 && j == 0) {
                              down2[i] = aux.right_to_left[{i, j + 1}].legs_rename({{TAT::legs_name::Up, leg_up2}});
                        } else if (j == peps.metadata.N - 1) {
                              down2[i] = down1[i];
                        } else {
                              down2[i] = Node::contract(
                                    down1[i],
                                    aux.right_to_left[{i, j + 1}].legs_rename({{TAT::legs_name::Up, leg_up2}}),
                                    {leg_up2},
                                    {TAT::legs_name::Down});
                        }

                        down3[i] = Node::contract(
                              down2[i],
                              lattice[{i, j}].legs_rename({{TAT::legs_name::Up, leg_up3}}),
                              {leg_up3, TAT::legs_name::Left, TAT::legs_name::Right},
                              {TAT::legs_name::Down, TAT::legs_name::Right, TAT::legs_name::Left});
                  }
            }
            for (int i = 0; i < peps.metadata.M; i++) {
                  if (i != peps.metadata.M - 1) {
                        for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                              auto tmp_up = up2[i].partial_contract(
                                    peps.lattice[{i, j, k}],
                                    {TAT::legs_name::Left, TAT::legs_name::Right, leg_down3},
                                    {TAT::legs_name::Right, TAT::legs_name::Left, TAT::legs_name::Up});
                              for (TAT::Size l = 0; l < peps.metadata.d; l++) {
                                    auto tmp_down = down2[i + 1].partial_contract(
                                          peps.lattice[{i + 1, j, l}],
                                          {TAT::legs_name::Left, TAT::legs_name::Right, leg_up3},
                                          {TAT::legs_name::Right, TAT::legs_name::Left, TAT::legs_name::Down});
                                    target.double_hole[{i, j, TAT::legs_name::Down, k, l}] = Node::contract(
                                          tmp_up,
                                          tmp_down,
                                          {leg_down1, leg_down2, TAT::legs_name::Down},
                                          {leg_up1, leg_up2, TAT::legs_name::Up});

                                    target.double_hole[{i + 1, j, TAT::legs_name::Up, l, k}] =
                                          target.double_hole[{i, j, TAT::legs_name::Down, k, l}];
                              }
                        }
                  }
            }
      }
}

#endif // TAT_USE_AUXILIARY_HPP_
