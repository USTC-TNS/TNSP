/* example/Heisenberg_PEPS_SU.dir/SVD_Graph_with_Env.hpp
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

#ifndef TAT_SVD_GRAPH_WITH_ENV_HPP_
#define TAT_SVD_GRAPH_WITH_ENV_HPP_

#include <TAT.hpp>

// 有环境的svd
template<template<class> class N, class Base, int a_env, int b_env>
struct SVD_Graph_with_Env {
      TAT::LazyNode<N, Base> old_A;
      TAT::LazyNode<N, Base> old_B;
      TAT::LazyNode<N, Base> old_env;

      std::array<TAT::LazyNode<N, Base>, a_env> A_env;
      std::array<TAT::LazyNode<N, Base>, b_env> B_env;

      TAT::LazyNode<N, Base> new_A;
      TAT::LazyNode<N, Base> new_B;
      TAT::LazyNode<N, Base> new_env;

      TAT::LazyNode<N, Base> H;

      SVD_Graph_with_Env(
            TAT::Legs a_leg,
            TAT::Legs b_leg,
            std::array<TAT::Legs, a_env> A_legs,
            std::array<TAT::Legs, b_env> B_legs,
            int cut) {
            using namespace TAT::legs_name;

            auto BigA = old_A;
            for (int i = 0; i < a_env; i++) {
                  BigA = BigA.multiple(A_env[i], A_legs[i]);
            }

            auto BigB = old_B;
            for (int i = 0; i < b_env; i++) {
                  BigB = BigB.multiple(B_env[i], B_legs[i]);
            }

            auto BigA_QR = BigA.rq({Phy, a_leg}, b_leg, a_leg);
            auto BigB_QR = BigB.rq({Phy, b_leg}, a_leg, b_leg);

            auto Big = BigA_QR.R.multiple(old_env, a_leg);
            Big = TAT::LazyNode<N, Base>::contract(
                  Big.legs_rename({{Phy, Phy1}}), BigB_QR.R.legs_rename({{Phy, Phy2}}), {a_leg}, {b_leg});
            Big = TAT::LazyNode<N, Base>::contract(Big, H, {Phy1, Phy2}, {Phy3, Phy4});
            Big = Big / Big.template norm<-1>();

            auto svd = Big.svd({Phy1, b_leg}, a_leg, b_leg, cut);
            new_env = svd.S;
            new_A = TAT::LazyNode<N, Base>::contract(svd.U, BigA_QR.Q, {b_leg}, {a_leg});
            new_B = TAT::LazyNode<N, Base>::contract(svd.V, BigB_QR.Q, {a_leg}, {b_leg});

            for (int i = 0; i < a_env; i++) {
                  new_A = new_A.multiple(1 / A_env[i], A_legs[i]);
            }
            for (int i = 0; i < b_env; i++) {
                  new_B = new_B.multiple(1 / B_env[i], B_legs[i]);
            }

            new_A = new_A.legs_rename({{Phy1, Phy}});
            new_B = new_B.legs_rename({{Phy2, Phy}});
      }
      void operator()(
            TAT::LazyNode<N, Base> H_value,
            TAT::LazyNode<N, Base> A,
            TAT::LazyNode<N, Base> B,
            TAT::LazyNode<N, Base> env,
            std::array<TAT::LazyNode<N, Base>, a_env> A_env_value,
            std::array<TAT::LazyNode<N, Base>, b_env> B_env_value) {
            old_A.set_value(A.pop());
            old_B.set_value(B.pop());
            old_env.set_value(env.pop());
            H.set_point_value(&H_value.value());

            for (int i = 0; i < a_env; i++) {
                  A_env[i].set_point_value(&A_env_value[i].value());
            }
            for (int i = 0; i < b_env; i++) {
                  B_env[i].set_point_value(&B_env_value[i].value());
            }

            A.set_value(new_A.pop());
            B.set_value(new_B.pop());
            env.set_value(new_env.pop());
      }
};

#endif // TAT_SVD_GRAPH_WITH_ENV_HPP_