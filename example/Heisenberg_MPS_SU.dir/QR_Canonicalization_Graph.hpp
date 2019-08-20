/* example/Heisenberg_MPS_SU.dir/QR_Canonicalization_Graph.hpp
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

#ifndef TAT_QR_CANONICALIZATION_GRAPH_HPP_
#define TAT_QR_CANONICALIZATION_GRAPH_HPP_

#include <TAT.hpp>

// QR正则化
template<template<class> class N, class Base>
struct QR_Canonicalization_Graph {
   public:
      TAT::LazyNode<N, Base> to_split;
      TAT::LazyNode<N, Base> to_absorb;
      TAT::LazyNode<N, Base> splited;
      TAT::LazyNode<N, Base> absorbed;
      QR_Canonicalization_Graph(TAT::Legs split_leg, TAT::Legs absorb_leg) {
            auto qr = to_split.rq({split_leg}, absorb_leg, split_leg);
            splited = qr.Q;
            absorbed = TAT::LazyNode<N, Base>::contract(qr.R, to_absorb, {split_leg}, {absorb_leg});
      }
      void operator()(TAT::LazyNode<N, Base> split, TAT::LazyNode<N, Base> absorb) {
            to_split.set_value(split.pop());
            to_absorb.set_value(absorb.pop());
            split.set_value(splited.pop());
            absorb.set_value(absorbed.pop());
      }
};

#endif // TAT_QR_CANONICALIZATION_GRAPH_HPP_