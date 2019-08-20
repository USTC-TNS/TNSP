/* example/Heisenberg_PEPS_GO.dir/Configuration.hpp
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

#ifndef TAT_CONFIGURATION_HPP_
#define TAT_CONFIGURATION_HPP_

#include <TAT.hpp>

#include "PEPS.hpp"

struct Configuration {
      using Node = TAT::LazyNode<TAT::Node, double>;

      PEPS_GO& peps;
      /**
       * 设置peps, 并初始化configuration的lattice
       */
      Configuration(PEPS_GO& peps) : peps(peps) {
            for (int i = 0; i < peps.metadata.M; i++) {
                  for (int j = 0; j < peps.metadata.N; j++) {
                        lattice[{i, j}].set_point_func(
                              [&peps, i, j](TAT::Size phy) {
                                    return &peps.lattice[{i, j, phy}];
                              },
                              state[{i, j}]);
                  }
            }
      }

      std::map<std::tuple<int, int>, TAT::Lazy<TAT::Size>> state;
      std::map<std::tuple<int, int>, Node> lattice;

      TAT::Size get_state(TAT::Size i, TAT::Size j) {
            return state[{i, j}].value();
      }

      /**
       * 四个方向的辅助矩阵
       */
      struct Auxiliary {
            std::map<std::tuple<int, int>, Node> up_to_down;
            std::map<std::tuple<int, int>, Node> down_to_up;
            std::map<std::tuple<int, int>, Node> left_to_right;
            std::map<std::tuple<int, int>, Node> right_to_left;
      };
      Auxiliary aux;

      // usable output node
      struct Target {
            std::map<std::tuple<int, int>, Node> gradient;
            std::map<std::tuple<int, int, TAT::Size>, Node> single_hole;
            std::map<std::tuple<int, int, TAT::Legs, TAT::Size, TAT::Size>, Node>
                  double_hole; // 每个点两个方right和down
      };
      Target target;

      const TAT::Node<double>& gradient(int i, int j) {
            return target.gradient[{i, j}].value();
      }

      const TAT::Node<double>& ws() {
            return target.single_hole[{0, 0, get_state(0, 0)}].value();
      }

      const TAT::Node<double>& single_hole(int i, int j, TAT::Size phy) {
            if (phy == get_state(i, j)) {
                  return ws();
            } else {
                  return target.single_hole[{i, j, phy}].value();
            }
      }

      const TAT::Node<double>& double_hole(int i, int j, TAT::Legs direction, TAT::Size phy1, TAT::Size phy2) {
            using namespace TAT::legs_name;
            int k = i;
            int l = j;
            if (direction == Left) {
                  l--;
            } else if (direction == Right) {
                  l++;
            } else if (direction == Up) {
                  k--;
            } else {
                  k++;
            }
            if (phy1 == get_state(i, j) && phy2 == get_state(k, l)) {
                  return ws();
            } else if (phy1 == get_state(i, j)) {
                  return target.single_hole[{k, l, phy2}].value();
            } else if (phy2 == get_state(k, l)) {
                  return target.single_hole[{i, j, phy1}].value();
            } else {
                  return target.double_hole[{i, j, direction, phy1, phy2}].value();
            }
      }

      void initial_aux(TAT::Size D_c, int qr_scan_time, std::function<double()> setter);

      /**
       * set single hole and double hole and ws
       */
      void calculate_hole();

      void set_state(const std::map<std::tuple<int, int>, TAT::Size>& st) {
            for (int i = 0; i < peps.metadata.M; i++) {
                  for (int j = 0; j < peps.metadata.N; j++) {
                        state[{i, j}].set_value(st.at({i, j}));
                  }
            }
      }
};

#include "create_Auxiliary.hpp"
#include "use_Auxiliary.hpp"

#endif // TAT_CONFIGURATION_HPP_
