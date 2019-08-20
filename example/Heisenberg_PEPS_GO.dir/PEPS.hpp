/* example/Heisenberg_PEPS_GO.dir/PEPS.hpp
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

#ifndef TAT_PEPS_HPP_
#define TAT_PEPS_HPP_

#include <TAT.hpp>

struct PEPS_GO {
      struct Metadata {
         public:
            int M;
            int N;
            TAT::Size d;
            TAT::Size D;
      };

      /**
       * 返回此点的legs, 不包含phy
       */
      auto legs_generator(int i, int j) {
            auto res = std::vector<TAT::Legs>{};
            if (i != 0) {
                  res.push_back(TAT::legs_name::Up);
            }
            if (i != metadata.M - 1) {
                  res.push_back(TAT::legs_name::Down);
            }
            if (j != 0) {
                  res.push_back(TAT::legs_name::Left);
            }
            if (j != metadata.N - 1) {
                  res.push_back(TAT::legs_name::Right);
            }
            return res;
      };

      /**
       * 返回此点的dims, 不包含phy, 如果后两个参数非空, 则将除了direction的legs设置为Dc而非D
       */
      auto dims_generator(int i, int j, TAT::Legs direction = TAT::Legs("placeholder_for_dim_gen"), TAT::Size D_c = 0) {
            if (direction == TAT::Legs("placeholder_for_dim_gen")) {
                  D_c = metadata.D;
            }
            auto res = std::vector<TAT::Size>{};
            if (i != 0) {
                  res.push_back(direction == TAT::legs_name::Up ? metadata.D : D_c);
            }
            if (i != metadata.M - 1) {
                  res.push_back(direction == TAT::legs_name::Down ? metadata.D : D_c);
            }
            if (j != 0) {
                  res.push_back(direction == TAT::legs_name::Left ? metadata.D : D_c);
            }
            if (j != metadata.N - 1) {
                  res.push_back(direction == TAT::legs_name::Right ? metadata.D : D_c);
            }
            return res;
      };

      Metadata metadata;
      std::map<std::tuple<int, int, TAT::Size>, TAT::Node<double>> lattice;

      /**
       * 初始化元数据
       */
      void initial_metadata(int M, int N, TAT::Size d, TAT::Size D) {
            metadata.M = M;
            metadata.N = N;
            metadata.d = d;
            metadata.D = D;
      }

      /**
       * 初始化lattice的形状
       */
      void initial_lattice() {
            for (int i = 0; i < metadata.M; i++) {
                  for (int j = 0; j < metadata.N; j++) {
                        for (TAT::Size k = 0; k < metadata.d; k++) {
                              lattice[{i, j, k}] = TAT::Node<double>(legs_generator(i, j), dims_generator(i, j));
                        }
                  }
            }
      }

      /**
       * 设置lattice的数据为随机数
       */
      void set_random_lattice(std::function<double()> setter) {
            for (int i = 0; i < metadata.M; i++) {
                  for (int j = 0; j < metadata.N; j++) {
                        for (TAT::Size k = 0; k < metadata.d; k++) {
                              lattice[{i, j, k}].set(setter);
                        }
                  }
            }
      }
};

#endif // TAT_PEPS_HPP_