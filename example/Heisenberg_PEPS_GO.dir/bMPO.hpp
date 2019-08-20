/* example/Heisenberg_PEPS_GO.dir/bMPO.hpp
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

#ifndef TAT_BMPO_HPP_
#define TAT_BMPO_HPP_

#include <TAT.hpp>

template<class Base>
using Default_Node_in_bMPO = TAT::LazyNode<TAT::Node, Base>;

template<template<class> class N = Default_Node_in_bMPO>
struct bounded_matrix_product_operator {
      using Node = N<double>;
      using HighNode = TAT::LazyNode<N, double>;

      struct bMPO_config {
            int length;
            int qr_scan_time;
            TAT::Legs left;
            TAT::Legs right;
            TAT::Legs up;
            TAT::Legs down;
            bool need_initial_canonicalization;
      };
      bMPO_config config;

      bounded_matrix_product_operator(const bMPO_config& config) : config(config){};

      // A-B- -> C-
      auto operator()( // up to down
            const std::vector<Node>& former_node,
            const std::vector<Node>& current_node,
            const std::vector<Node>& initial_node) {
            // create graph
            std::vector<HighNode> former(config.length);
            std::vector<HighNode> current(config.length);
            std::vector<HighNode> res(config.length);
            for (int i = 0; i < config.length; i++) {
                  former[i].set_point_value(&former_node[i]);
                  current[i].set_point_value(&current_node[i]);
                  res[i].set_point_value(&initial_node[i]);
            }
            // canonicalization except 0
            if (config.need_initial_canonicalization) {
                  for (int i = config.length - 1; i > 0; i--) {
                        auto tmp = res[i].rq({config.left}, config.right, config.left);
                        res[i] = tmp.Q;
                        res[i - 1] = HighNode::contract(res[i - 1], tmp.R, {config.right}, {config.left});
                        // qr.R could be dropped, but not, since if drop, need cut in QR
                  }
            }
            // define graph
            std::vector<HighNode> left1(config.length);
            std::vector<HighNode> left2(config.length);
            std::vector<HighNode> left3(config.length);
            auto leg_right1 = TAT::Legs("bMPO_right1");
            auto leg_right2 = TAT::Legs("bMPO_right2");
            auto leg_right3 = TAT::Legs("bMPO_right3");
            for (int i = 0; i < config.length; i++) {
                  if (i == 0) {
                        left1[i] = former[i].legs_rename({{config.right, leg_right1}});
                  } else {
                        left1[i] = HighNode::contract(
                              left3[i - 1],
                              former[i].legs_rename({{config.right, leg_right1}}),
                              {leg_right1},
                              {config.left});
                  }
                  left2[i] = HighNode::contract(
                        left1[i],
                        current[i].legs_rename({{config.right, leg_right2}}),
                        {config.down, leg_right2},
                        {config.up, config.left});
                  left3[i] = HighNode::contract(
                        left2[i],
                        res[i].legs_rename({{config.right, leg_right3}}),
                        {config.down, leg_right3},
                        {config.down, config.left});
            }
            std::vector<HighNode> right1(config.length);
            std::vector<HighNode> right2(config.length);
            std::vector<HighNode> right3(config.length);
            auto leg_left1 = TAT::Legs("bMPO_left1");
            auto leg_left2 = TAT::Legs("bMPO_left2");
            auto leg_left3 = TAT::Legs("bMPO_left3");
            for (int i = config.length - 1; i >= 0; i--) {
                  if (i == config.length - 1) {
                        right1[i] = former[i].legs_rename({{config.left, leg_left1}});
                  } else {
                        right1[i] = HighNode::contract(
                              right3[i + 1],
                              former[i].legs_rename({{config.left, leg_left1}}),
                              {leg_left1},
                              {config.right});
                  }
                  right2[i] = HighNode::contract(
                        right1[i],
                        current[i].legs_rename({{config.left, leg_left2}}),
                        {config.down, leg_left2},
                        {config.up, config.right});
                  right3[i] = HighNode::contract(
                        right2[i],
                        res[i].legs_rename({{config.left, leg_left3}}),
                        {config.down, leg_left3},
                        {config.down, config.right});
            }
            std::vector<HighNode> target(config.length);
            for (int i = 0; i < config.length; i++) {
                  if (i == 0) {
                        target[i] = right2[i].legs_rename({{leg_left3, config.right}});
                  } else if (i == config.length - 1) {
                        target[i] = left2[i].legs_rename({{leg_right3, config.left}});
                  } else {
                        target[i] = HighNode::contract(
                                          left2[i], right3[i + 1], {leg_right1, leg_right2}, {leg_left1, leg_left2})
                                          .legs_rename({{leg_right3, config.left}, {leg_left3, config.right}});
                  }
            }
            // scan
            for (int i = 0; i < config.qr_scan_time; i++) {
                  // status: canonical except 0
                  for (int j = 0; j < config.length - 1; j++) {
                        res[j].set_value(target[j].pop());
                        res[j].set_value(res[j].pop().rq({config.right}, config.left, config.right).Q);
                        // R dropped
                  }
                  // status: canonical except L-1
                  for (int j = config.length - 1; j > 0; j--) {
                        res[j].set_value(target[j].pop());
                        res[j].set_value(res[j].pop().rq({config.left}, config.right, config.left).Q);
                        // R dropped
                  }
            }
            res[0].set_value(target[0].pop());
            std::vector<Node> res_node(config.length);
            for (int i = 0; i < config.length; i++) {
                  res_node[i] = res[i].pop();
            }
            return res_node;
      }
};

#endif // TAT_BMPO_HPP_
