/**
 * \file example/PEPS_GO.dir/create_Auxiliary.hpp
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

#ifndef TAT_CREATE_AUXILIARY_HPP_
#define TAT_CREATE_AUXILIARY_HPP_

#include "Configuration.hpp"

#include "bMPO.hpp"

void Configuration::initial_aux(TAT::Size D_c, int qr_scan_time, std::function<double()> setter) {
      // up to down
      {
            auto bMPO = bounded_matrix_product_operator<>({peps.metadata.N,
                                                           qr_scan_time,
                                                           TAT::legs_name::Left,
                                                           TAT::legs_name::Right,
                                                           TAT::legs_name::Up,
                                                           TAT::legs_name::Down,
                                                           true});
            for (int i = 0; i < peps.metadata.M - 1; i++) {
                  if (i == 0) {
                        for (int j = 0; j < peps.metadata.N; j++) {
                              aux.up_to_down[{i, j}] = lattice[{i, j}];
                        }
                  } else {
                        auto tmp_lattice = std::vector<Node>(peps.metadata.N);
                        auto tmp_aux = std::vector<Node>(peps.metadata.N);
                        auto initial = std::vector<Node>(peps.metadata.N);
                        for (int j = 0; j < peps.metadata.N; j++) {
                              tmp_lattice[j] = lattice[{i, j}];
                              tmp_aux[j] = aux.up_to_down[{i - 1, j}];
                              initial[j] =
                                    Node(peps.legs_generator(0, j),
                                         peps.dims_generator(0, j, TAT::legs_name::Down, D_c)); // use D_c
                              initial[j].set(setter); //
                        }
                        auto next_aux = bMPO(tmp_aux, tmp_lattice, initial);
                        for (int j = 0; j < peps.metadata.N; j++) {
                              aux.up_to_down[{i, j}] = next_aux[j];
                        }
                  }
            }
      }
      // down to up
      {
            auto bMPO = bounded_matrix_product_operator<>({peps.metadata.N,
                                                           qr_scan_time,
                                                           TAT::legs_name::Left,
                                                           TAT::legs_name::Right,
                                                           TAT::legs_name::Down,
                                                           TAT::legs_name::Up,
                                                           true});
            for (int i = peps.metadata.M - 1; i >= 0 + 1; i--) {
                  if (i == peps.metadata.M - 1) {
                        for (int j = 0; j < peps.metadata.N; j++) {
                              aux.down_to_up[{i, j}] = lattice[{i, j}];
                        }
                  } else {
                        auto tmp_lattice = std::vector<Node>(peps.metadata.N);
                        auto tmp_aux = std::vector<Node>(peps.metadata.N);
                        auto initial = std::vector<Node>(peps.metadata.N);
                        for (int j = 0; j < peps.metadata.N; j++) {
                              tmp_lattice[j] = lattice[{i, j}];
                              tmp_aux[j] = aux.down_to_up[{i + 1, j}];
                              initial[j] = Node(
                                    peps.legs_generator(peps.metadata.M - 1, j),
                                    peps.dims_generator(peps.metadata.M - 1, j, TAT::legs_name::Up, D_c)); // use D_c
                              initial[j].set(setter); //
                        }
                        auto next_aux = bMPO(tmp_aux, tmp_lattice, initial);
                        for (int j = 0; j < peps.metadata.N; j++) {
                              aux.down_to_up[{i, j}] = next_aux[j];
                        }
                  }
            }
      }
      // left to right
      {
            auto bMPO = bounded_matrix_product_operator<>({peps.metadata.M,
                                                           qr_scan_time,
                                                           TAT::legs_name::Up,
                                                           TAT::legs_name::Down,
                                                           TAT::legs_name::Left,
                                                           TAT::legs_name::Right,
                                                           true});
            for (int j = 0; j < peps.metadata.N - 1; j++) {
                  if (j == 0) {
                        for (int i = 0; i < peps.metadata.M; i++) {
                              aux.left_to_right[{i, j}] = lattice[{i, j}];
                        }
                  } else {
                        auto tmp_lattice = std::vector<Node>(peps.metadata.M);
                        auto tmp_aux = std::vector<Node>(peps.metadata.M);
                        auto initial = std::vector<Node>(peps.metadata.M);
                        for (int i = 0; i < peps.metadata.M; i++) {
                              tmp_lattice[i] = lattice[{i, j}];
                              tmp_aux[i] = aux.left_to_right[{i, j - 1}];
                              initial[i] =
                                    Node(peps.legs_generator(i, 0),
                                         peps.dims_generator(i, 0, TAT::legs_name::Right, D_c)); // use D_c
                              initial[i].set(setter); //
                        }
                        auto next_aux = bMPO(tmp_aux, tmp_lattice, initial);
                        for (int i = 0; i < peps.metadata.M; i++) {
                              aux.left_to_right[{i, j}] = next_aux[i];
                        }
                  }
            }
      }
      // right to left
      {
            auto bMPO = bounded_matrix_product_operator<>({peps.metadata.M,
                                                           qr_scan_time,
                                                           TAT::legs_name::Up,
                                                           TAT::legs_name::Down,
                                                           TAT::legs_name::Right,
                                                           TAT::legs_name::Left,
                                                           true});
            for (int j = peps.metadata.N - 1; j >= 0 + 1; j--) {
                  if (j == peps.metadata.N - 1) {
                        for (int i = 0; i < peps.metadata.M; i++) {
                              aux.right_to_left[{i, j}] = lattice[{i, j}];
                        }
                  } else {
                        auto tmp_lattice = std::vector<Node>(peps.metadata.M);
                        auto tmp_aux = std::vector<Node>(peps.metadata.M);
                        auto initial = std::vector<Node>(peps.metadata.M);
                        for (int i = 0; i < peps.metadata.M; i++) {
                              tmp_lattice[i] = lattice[{i, j}];
                              tmp_aux[i] = aux.right_to_left[{i, j + 1}];
                              initial[i] = Node(
                                    peps.legs_generator(i, peps.metadata.N - 1),
                                    peps.dims_generator(i, peps.metadata.N - 1, TAT::legs_name::Left, D_c)); // use D_c
                              initial[i].set(setter); //
                        }
                        auto next_aux = bMPO(tmp_aux, tmp_lattice, initial);
                        for (int i = 0; i < peps.metadata.M; i++) {
                              aux.right_to_left[{i, j}] = next_aux[i];
                        }
                  }
            }
      }
}

#endif // TAT_CREATE_AUXILIARY_HPP_
