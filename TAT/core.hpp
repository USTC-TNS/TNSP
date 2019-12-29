#pragma once
#ifndef TAT_CORE_HPP_
#   define TAT_CORE_HPP_

#   include "edge.hpp"

namespace TAT {
   template<class ScalarType, class Symmetry>
   struct Block {
      vector<Symmetry> symmetries;
      vector<ScalarType> raw_data;
      Size size;

      template<
            class T = vector<Symmetry>,
            class = std::enable_if_t<is_same_nocvref_v<T, vector<Symmetry>>>>
      Block(const vector<Edge<Symmetry>>& e, T&& s) : symmetries(std::forward<T>(s)) {
         size = 1;
         for (Rank i = 0; i < e.size(); i++) {
            size *= e[i].at(symmetries[i]);
         }
         raw_data = vector<ScalarType>(size);
      }
   };

   template<class Symmetry>
   auto initialize_block_symmetries_with_check(const vector<Edge<Symmetry>>& edges) {
      auto res = vector<vector<Symmetry>>();
      auto vec = vector<Symmetry>();
      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;
      loop_edge(
            edges,
            [&]() { res.push_back({}); },
            [&](const PosType& pos) {
               for (const auto& i : pos) {
                  vec.push_back(i->first);
               }
            },
            [&]([[maybe_unused]] const PosType& pos) {
               auto sum = Symmetry();
               for (const auto& i : vec) {
                  sum += i;
               }
               return sum == Symmetry();
            },
            [&]([[maybe_unused]] const PosType& pos) { res.push_back(vec); },
            [&](const PosType& pos, Rank ptr) {
               for (Rank i = ptr; i < pos.size(); i++) {
                  vec[i] = pos[i]->first;
               }
            });
      return res;
   }

   template<class ScalarType, class Symmetry>
   struct Core {
      vector<Edge<Symmetry>> edges;
      vector<Block<ScalarType, Symmetry>> blocks;

      template<
            class T = vector<Edge<Symmetry>>,
            class = std::enable_if_t<is_same_nocvref_v<T, vector<Edge<Symmetry>>>>>
      Core(T&& e) : edges(std::forward<T>(e)) {
         auto symmetries_list = initialize_block_symmetries_with_check(edges);
         for (auto& i : symmetries_list) {
            blocks.push_back(Block<ScalarType, Symmetry>(edges, std::move(i)));
         }
      }

      using PosType = vector<typename Edge<Symmetry>::const_iterator>;

      auto find_block(const vector<Symmetry>& syms) {
         Rank rank = Rank(edges.size());
         Nums number = Nums(blocks.size());
         for (Nums i = 0; i<number; i++) {
            if (syms == blocks[i].symmetries) {
               return i;
            }
         }
         TAT_WARNING("Block Not Found");
         return number;
      }
   };
} // namespace TAT
#endif
