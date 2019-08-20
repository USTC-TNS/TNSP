/* example/Heisenberg_MPS_SU.dir/SVD_Graph_without_Env.hpp
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

#ifndef TAT_SVD_GRAPH_WITHOUT_ENV_HPP_
#define TAT_SVD_GRAPH_WITHOUT_ENV_HPP_

#include <TAT.hpp>

// 无环境的svd
template<template<class> class N, class Base>
struct SVD_Graph_without_Env {
      TAT::LazyNode<N, Base> old_A;
      TAT::LazyNode<N, Base> old_B;
      TAT::LazyNode<N, Base> new_A;
      TAT::LazyNode<N, Base> new_B;
      SVD_Graph_without_Env(TAT::LazyNode<N, Base> H, int cut, bool left = true) {
            using namespace TAT::legs_name;
            auto big = TAT::LazyNode<N, Base>::contract(
                  old_A.legs_rename({{Phy, Phy1}}), old_B.legs_rename({{Phy, Phy2}}), {Right}, {Left});
            auto Big = TAT::LazyNode<N, Base>::contract(big, H, {Phy1, Phy2}, {Phy3, Phy4});
            Big = Big / Big.template norm<-1>();
            auto svd = Big.svd({Phy1, Left}, Right, Left, cut);
            new_A = svd.U.legs_rename({{Phy1, Phy}});
            new_B = svd.V.legs_rename({{Phy2, Phy}});
            if (left) {
                  new_A = new_A.multiple(svd.S, Right);
            } else {
                  new_B = new_B.multiple(svd.S, Left);
            }
      }
      void operator()(TAT::LazyNode<N, Base> A, TAT::LazyNode<N, Base> B) {
            old_A.set_value(A.pop());
            old_B.set_value(B.pop());
            A.set_value(new_A.pop());
            B.set_value(new_B.pop());
      }
};

#endif // TAT_SVD_GRAPH_WITHOUT_ENV_HPP_