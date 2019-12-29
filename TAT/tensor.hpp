#pragma once
#ifndef TAT_TENSOR_HPP_
#   define TAT_TENSOR_HPP_

#   include <algorithm>
#   include <cassert>
#   include <cmath>
#   include <complex>
#   include <cstring>
#   include <iostream>
#   include <map>
#   include <memory>
#   include <optional>
#   include <set>
#   include <string>
#   include <type_traits>
#   include <variant>
#   include <vector>

#   include "core.hpp"
#   include "edge.hpp"
#   include "misc.hpp"
#   include "name.hpp"
#   include "symmetry.hpp"

// #include <mpi.h>
// TODO: MPI 这个最后弄, 不然valgrind一大堆报错

namespace TAT {
   std::map<Name, Rank> construct_name_to_index(const vector<Name>& names) {
      std::map<Name, Rank> res;
      auto rank = Rank(names.size());
      for (Rank i = 0; i < rank; i++) {
         res[names[i]] = i;
      }
      return res;
   }

   bool is_valid_name(const vector<Name>& names, const Rank& rank) {
      return names.size() == std::set<Name>(names.begin(), names.end()).size() &&
             names.size() == rank;
   }

   template<class ScalarType = double, class Symmetry = NoSymmetry>
   struct Tensor {
      // initialize
      vector<Name> names;
      std::map<Name, Rank> name_to_index;
      std::shared_ptr<Core<ScalarType, Symmetry>> core;

      template<
            class U = vector<Name>,
            class T = vector<Edge<Symmetry>>,
            class = std::enable_if_t<is_same_nocvref_v<U, vector<Name>>>,
            class = std::enable_if_t<is_same_nocvref_v<T, vector<Edge<Symmetry>>>>>
      Tensor(U&& n, T&& e) :
            names(std::forward<U>(n)), name_to_index(construct_name_to_index(names)),
            core(std::make_shared<Core<ScalarType, Symmetry>>(std::forward<T>(e))) {
         if (!is_valid_name(names, core->edges.size())) {
            TAT_WARNING("Invalid Names");
         }
      }

      Tensor() = default;
      Tensor(const Tensor& other) :
            names(other.names), name_to_index(other.name_to_index),
            core(std::make_shared<Core<ScalarType, Symmetry>>(*other.core)) {
         TAT_WARNING("Data Copy In Tensor Copy");
      }
      Tensor(Tensor&& other) = default;
      ~Tensor() = default;
      Tensor& operator=(const Tensor& other) {
         names = other.names;
         name_to_index = other.name_to_index;
         core = std::make_shared<Core<ScalarType, Symmetry>>(*other.core);
         TAT_WARNING("Data Copy In Tensor Copy");
      }
      Tensor& operator=(Tensor&& other) = default;

      Tensor(ScalarType num) : Tensor({}, {}) {
         core->blocks[0].raw_data[0] = num;
      }

      operator ScalarType() const {
         if (names.size() != 0) {
            TAT_WARNING("Conversion From Multiple Rank Tensor To Scalar");
         }
         return core->blocks[0].raw_data[0];
      }

      template<class Generator>
      Tensor<ScalarType, Symmetry>& set(Generator&& generator) & {
         if (core.use_count() != 1) {
            TAT_WARNING("Set Tensor Shared");
         }
         for (auto& i : core->blocks) {
            std::generate(i.raw_data.begin(), i.raw_data.begin() + i.size, generator);
         }
         return *this;
      }
      template<class Generator>
      Tensor<ScalarType, Symmetry> set(Generator&& generator) && {
         return std::move(set(generator));
      }

      Tensor<ScalarType, Symmetry>& zero() & {
         return set([]() { return 0; });
      }
      Tensor<ScalarType, Symmetry> zero() && {
         return std::move(zero());
      }

      // get item
   private:
      std::tuple<Nums, Size>
      get_pos_for_at(const std::map<Name, EdgePosition<Symmetry>>& position) const {
         auto rank = Rank(names.size());
         vector<Symmetry> block_symmetries(rank);
         vector<Size> scalar_position(rank);
         vector<Size> dimensions(rank);
         for (const auto& [name, res] : position) {
            auto index = name_to_index.at(name);
            block_symmetries[index] = res.sym;
            scalar_position[index] = res.position;
            dimensions[index] = core->edges[index].at(res.sym);
         }
         Size offset = 0;
         for (Rank j = 0; j < rank; j++) {
            offset *= dimensions[j];
            offset += scalar_position[j];
         }
         for (Nums i = 0; i < core->blocks.size(); i++) {
            if (block_symmetries == core->blocks[i].symmetries) {
               return {i, offset};
            }
         }
         TAT_WARNING("Cannot Find Correct Block When Get Item");
         return {0, 0};
      }

   public:
      const ScalarType& at(const std::map<Name, EdgePosition<Symmetry>>& position) const& {
         auto [sym, pos] = get_pos_for_at(position);
         return core->blocks[sym].raw_data[pos];
      }

      ScalarType& at(const std::map<Name, EdgePosition<Symmetry>>& position) & {
         auto [sym, pos] = get_pos_for_at(position);
         return core->blocks[sym].raw_data[pos];
      }

      ScalarType at(const std::map<Name, EdgePosition<Symmetry>>& position) && {
         auto [sym, pos] = get_pos_for_at(position);
         return core->blocks[sym].raw_data[pos];
      }

      // conversion
      template<class OtherScalarType>
      Tensor<OtherScalarType, Symmetry> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            auto res = Tensor<ScalarType, Symmetry>{};
            res.names = names;
            res.name_to_index = name_to_index;
            res.core = core;
            return res;
         } else {
            auto res = Tensor<OtherScalarType, Symmetry>{};
            res.names = names;
            res.name_to_index = name_to_index;
            res.core = std::make_shared<Core<OtherScalarType, Symmetry>>(core->edges);
            auto blocks_num = Nums(core->blocks.size());
            for (Nums i = 0; i < blocks_num; i++) {
               Size block_size = core->blocks[i].size;
               auto& dst = res.core->blocks[i].raw_data;
               const auto& src = core->blocks[i].raw_data;
               for (Size j = 0; j < block_size; j++) {
                  dst[j] = static_cast<OtherScalarType>(src[j]);
               }
            }
            return res;
         }
      }

      // norm
      template<int p = 2>
      Tensor<real_base_t<ScalarType>, Symmetry> norm() const {
         real_base_t<ScalarType> res = 0;
         if constexpr (p == -1) {
            auto blocks_num = Nums(core->blocks.size());
            for (Nums i = 0; i < blocks_num; i++) {
               const auto& block = core->blocks[i];
               const auto& data = block.raw_data;
               for (Size j = 0; j < block.size; j++) {
                  auto tmp = std::abs(data[j]);
                  if (tmp > res) {
                     res = tmp;
                  }
               }
            }
         } else if constexpr (p == 0) {
            auto blocks_num = Nums(core->blocks.size());
            for (Nums i = 0; i < blocks_num; i++) {
               res += real_base_t<ScalarType>(core->blocks[i].size);
            }
         } else {
            auto blocks_num = Nums(core->blocks.size());
            for (Nums i = 0; i < blocks_num; i++) {
               const auto& block = core->blocks[i];
               const auto& data = block.raw_data;
               for (Size j = 0; j < block.size; j++) {
                  if constexpr (p == 1) {
                     res += std::abs(data[j]);
                  } else if constexpr (p == 2) {
                     if constexpr (std::is_same_v<ScalarType, real_base_t<ScalarType>>) {
                        auto tmp = data[j];
                        res += tmp * tmp;
                     } else {
                        auto tmp = std::abs(data[j]);
                        res += tmp * tmp;
                     }
                  } else {
                     if constexpr (
                           p % 2 == 0 && std::is_same_v<ScalarType, real_base_t<ScalarType>>) {
                        res += std::pow(data[j], p);
                     } else {
                        res += std::pow(std::abs(data[j]), p);
                     }
                  }
               }
            }
            return res = std::pow(res, 1. / p);
         }
         return res;
      }

      // edge rename
      Tensor<ScalarType, Symmetry> edge_rename(const std::map<Name, Name>& dict) const {
         auto res = Tensor<ScalarType, Symmetry>{};
         res.core = core;
         std::transform(
               names.begin(), names.end(), std::back_inserter(res.names), [&dict](Name name) {
                  auto pos = dict.find(name);
                  if (pos == dict.end()) {
                     return name;
                  } else {
                     return pos->second;
                  }
               });
         res.name_to_index = construct_name_to_index(res.names);
         return res;
      }

      // TODO: 开始混乱了， 需要整理
   private:
      template<bool parity>
      static void matrix_transpose_kernel(
            Size M,
            Size N,
            const void* __restrict src,
            Size leading_src,
            void* __restrict dst,
            Size leading_dst,
            Size scalar_size) {
         Size line_size = scalar_size / sizeof(ScalarType);
         for (Size i = 0; i < M; i++) {
            for (Size j = 0; j < N; j++) {
               ScalarType* line_dst =
                     (ScalarType*)((char*)dst + (j * leading_dst + i) * scalar_size);
               const ScalarType* line_src =
                     (ScalarType*)((char*)src + (i * leading_src + j) * scalar_size);
               for (Size k = 0; k < line_size; k++) {
                  if constexpr (parity) {
                     line_dst[k] = -line_src[k];
                  } else {
                     line_dst[k] = line_src[k];
                  }
               }
               // TODO: 向量化
            }
         }
      }

      template<bool parity, Size cache_size, Size... other>
      static void matrix_transpose(
            Size M,
            Size N,
            const void* __restrict src,
            Size leading_src,
            void* __restrict dst,
            Size leading_dst,
            Size scalar_size) {
         Size block_size = 1;
         // TODO: 是否应该乘以二做冗余？
         while (block_size * block_size * scalar_size * 2 < cache_size) {
            block_size <<= 1;
         }
         block_size >>= 1;
         for (Size i = 0; i < M; i += block_size) {
            for (Size j = 0; j < N; j += block_size) {
               void* block_dst = (char*)dst + (j * leading_dst + i) * scalar_size;
               const void* block_src = (char*)src + (i * leading_src + j) * scalar_size;
               Size m = M - i;
               Size n = N - j;
               m = (block_size <= m) ? block_size : m;
               n = (block_size <= n) ? block_size : n;

               if constexpr (sizeof...(other) == 0) {
                  matrix_transpose_kernel<parity>(
                        m, n, block_src, leading_src, block_dst, leading_dst, scalar_size);
               } else {
                  matrix_transpose<parity, other...>(
                        m, n, block_src, leading_src, block_dst, leading_dst, scalar_size);
               }
            }
         }
      }

      static const Size l1_cache = 32768;
      static const Size l2_cache = 262144;
      static const Size l3_cache = 9437184;
      // TODO: 如何确定系统cache

      template<bool parity>
      static void block_transpose(
            const void* __restrict src,
            void* __restrict dst,
            const vector<Rank>& plan_src_to_dst,
            const vector<Rank>& plan_dst_to_src,
            const vector<Size>& dims_src,
            const vector<Size>& dims_dst,
            [[maybe_unused]] const Size& size,
            const Rank& rank,
            const Size& scalar_size) {
         vector<Size> step_src(rank);
         step_src[rank - 1] = 1;
         for (Rank i = rank - 1; i > 0; i--) {
            step_src[i - 1] = step_src[i] * dims_src[i];
         }
         vector<Size> step_dst(rank);
         step_dst[rank - 1] = 1;
         for (Rank i = rank - 1; i > 0; i--) {
            step_dst[i - 1] = step_dst[i] * dims_dst[i];
         }

         vector<Size> index_list_src(rank, 0);
         vector<Size> index_list_dst(rank, 0);
         Size index_src = 0;
         Size index_dst = 0;

         Size dim_N = dims_src[rank - 1];
         Size dim_M = dims_dst[rank - 1];
         Rank pos_M = plan_dst_to_src[rank - 1];
         Rank pos_N = plan_src_to_dst[rank - 1];
         Size leading_M = step_src[pos_M];
         Size leading_N = step_dst[pos_N];

         while (1) {
            // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
            matrix_transpose<parity, l2_cache, l1_cache>(
                  dim_M,
                  dim_N,
                  (char*)src + index_src * scalar_size,
                  leading_M,
                  (char*)dst + index_dst * scalar_size,
                  leading_N,
                  scalar_size);

            Rank temp_rank_dst = rank - 2;
            Rank temp_rank_src = plan_dst_to_src[temp_rank_dst];
            if (temp_rank_src == rank - 1) {
               if (temp_rank_dst == 0) {
                  return;
               }
               temp_rank_dst -= 1;
               temp_rank_src = plan_dst_to_src[temp_rank_dst];
            }

            index_list_src[temp_rank_src] += 1;
            index_src += step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] += 1;
            index_dst += step_dst[temp_rank_dst];

            while (index_list_dst[temp_rank_dst] == dims_dst[temp_rank_dst]) {
               index_list_src[temp_rank_src] = 0;
               index_src -= dims_src[temp_rank_src] * step_src[temp_rank_src];
               index_list_dst[temp_rank_dst] = 0;
               index_dst -= dims_dst[temp_rank_dst] * step_dst[temp_rank_dst];
               if (temp_rank_dst == 0) {
                  return;
               }
               temp_rank_dst -= 1;
               temp_rank_src = plan_dst_to_src[temp_rank_dst];
               if (temp_rank_src == rank - 1) {
                  if (temp_rank_dst == 0) {
                     return;
                  }
                  temp_rank_dst -= 1;
                  temp_rank_src = plan_dst_to_src[temp_rank_dst];
               }
               index_list_src[temp_rank_src] += 1;
               index_src += step_src[temp_rank_src];
               index_list_dst[temp_rank_dst] += 1;
               index_dst += step_dst[temp_rank_dst];
            }
         }
      }

      static auto noone_in_transpose(
            const vector<Rank>& plan_src_to_dst,
            const vector<Rank>& plan_dst_to_src,
            const vector<Size>& dims_src,
            const vector<Size>& dims_dst,
            const Rank& rank) {
         vector<bool> isone_src(rank);
         vector<bool> isone_dst(rank);
         for (Rank i = 0; i < rank; i++) {
            isone_src[i] = dims_src[i] == 1;
         }
         for (Rank i = 0; i < rank; i++) {
            isone_dst[i] = dims_dst[i] == 1;
         }
         vector<Rank> accum_src(rank);
         vector<Rank> accum_dst(rank);
         accum_src[0] = isone_src[0];
         for (Rank i = 1; i < rank; i++) {
            accum_src[i] = accum_src[i - 1] + Rank(isone_src[i]);
         }
         accum_dst[0] = isone_dst[0];
         for (Rank i = 1; i < rank; i++) {
            accum_dst[i] = accum_dst[i - 1] + Rank(isone_dst[i]);
         }

         vector<Rank> noone_plan_src_to_dst;
         vector<Rank> noone_plan_dst_to_src;
         for (Rank i = 0; i < rank; i++) {
            if (!isone_src[i]) {
               noone_plan_src_to_dst.push_back(plan_src_to_dst[i] - accum_dst[plan_src_to_dst[i]]);
            }
         }
         for (Rank i = 0; i < rank; i++) {
            if (!isone_dst[i]) {
               noone_plan_dst_to_src.push_back(plan_dst_to_src[i] - accum_src[plan_dst_to_src[i]]);
            }
         }
         auto noone_rank = Rank(noone_plan_dst_to_src.size());

         vector<Size> noone_dims_src;
         vector<Size> noone_dims_dst;
         for (Rank i = 0; i < rank; i++) {
            if (dims_src[i] != 1) {
               noone_dims_src.push_back(dims_src[i]);
            }
         }
         for (Rank i = 0; i < rank; i++) {
            if (dims_dst[i] != 1) {
               noone_dims_dst.push_back(dims_dst[i]);
            }
         }
         return std::tuple{noone_plan_src_to_dst,
                           noone_plan_dst_to_src,
                           noone_dims_src,
                           noone_dims_dst,
                           noone_rank};
      }

      static auto merge_in_transpose(
            const vector<Rank>& plan_src_to_dst,
            const vector<Rank>& plan_dst_to_src,
            const vector<Size>& dims_src,
            const vector<Size>& dims_dst,
            const Rank& rank) {
         vector<bool> merged_src_to_dst(rank);
         vector<bool> merged_dst_to_src(rank);
         merged_src_to_dst[0] = false;
         for (Rank i = 1; i < rank; i++) {
            if (plan_src_to_dst[i] == plan_src_to_dst[i - 1] + 1) {
               merged_src_to_dst[i] = true;
            } else {
               merged_src_to_dst[i] = false;
            }
         }
         merged_dst_to_src[0] = false;
         for (Rank i = 1; i < rank; i++) {
            if (plan_dst_to_src[i] == plan_dst_to_src[i - 1] + 1) {
               merged_dst_to_src[i] = true;
            } else {
               merged_dst_to_src[i] = false;
            }
         }
         vector<Rank> accum_src_to_dst(rank);
         vector<Rank> accum_dst_to_src(rank);
         accum_src_to_dst[0] = 0;
         for (Rank i = 1; i < rank; i++) {
            accum_src_to_dst[i] = accum_src_to_dst[i - 1] + Rank(merged_src_to_dst[i]);
         }
         accum_dst_to_src[0] = 0;
         for (Rank i = 1; i < rank; i++) {
            accum_dst_to_src[i] = accum_dst_to_src[i - 1] + Rank(merged_dst_to_src[i]);
         }
         vector<Rank> merged_plan_src_to_dst;
         vector<Rank> merged_plan_dst_to_src;
         for (Rank i = 0; i < rank; i++) {
            if (!merged_src_to_dst[i]) {
               merged_plan_src_to_dst.push_back(
                     plan_src_to_dst[i] - accum_dst_to_src[plan_src_to_dst[i]]);
            }
         }
         for (Rank i = 0; i < rank; i++) {
            if (!merged_dst_to_src[i]) {
               merged_plan_dst_to_src.push_back(
                     plan_dst_to_src[i] - accum_src_to_dst[plan_dst_to_src[i]]);
            }
         }
         auto merged_rank = Rank(merged_plan_src_to_dst.size());
         vector<Size> merged_dims_src(merged_rank);
         vector<Size> merged_dims_dst(merged_rank);
         Rank tmp_src_index = rank;
         for (Rank i = merged_rank; i-- > 0;) {
            merged_dims_src[i] = dims_src[--tmp_src_index];
            while (merged_src_to_dst[tmp_src_index]) {
               merged_dims_src[i] *= dims_src[--tmp_src_index];
            }
         }
         Rank tmp_dst_index = rank;
         for (Rank i = merged_rank; i-- > 0;) {
            merged_dims_dst[i] = dims_dst[--tmp_dst_index];
            while (merged_dst_to_src[tmp_dst_index]) {
               merged_dims_dst[i] *= dims_dst[--tmp_dst_index];
            }
         }
         return std::tuple{merged_plan_src_to_dst,
                           merged_plan_dst_to_src,
                           merged_dims_src,
                           merged_dims_dst,
                           merged_rank};
      }

      static bool check_same_symmetries(
            const vector<Symmetry>& s1,
            const vector<Symmetry>& s2,
            const vector<Rank>& plan) {
         auto rank = Rank(plan.size());
         for (Rank i = 0; i < rank; i++) {
            if (s1[i] != s2[plan[i]]) {
               return false;
            }
         }
         return true;
      }

   public:
      template<class T = vector<Name>, class = std::enable_if_t<is_same_nocvref_v<T, vector<Name>>>>
      Tensor<ScalarType, Symmetry> transpose(T&& target_names) const {
         return edge_operator({}, {}, {}, std::forward<T>(target_names));
         // bool parity = Symmetry::get_parity(core->blocks[index_src].symmetries, plan_src_to_dst);
         // TODO: 需要考虑merge parity， 转置parity 和 split parity
      }

      static void do_transpose(
            const vector<Rank>& plan_src_to_dst,
            const vector<Rank>& plan_dst_to_src,
            const ScalarType* src_data,
            ScalarType* dst_data,
            const vector<Size>& dims_src,
            const vector<Size>& dims_dst,
            Size block_size,
            Rank rank,
            bool parity) {
         if (block_size == 1) {
            if (parity) {
               *dst_data = -*src_data;
            } else {
               *dst_data = *src_data;
            }
            return;
         }

         auto [noone_plan_src_to_dst,
               noone_plan_dst_to_src,
               noone_dims_src,
               noone_dims_dst,
               noone_rank] =
               noone_in_transpose(plan_src_to_dst, plan_dst_to_src, dims_src, dims_dst, rank);

         auto [noone_merged_plan_src_to_dst,
               noone_merged_plan_dst_to_src,
               noone_merged_dims_src,
               noone_merged_dims_dst,
               noone_merged_rank] =
               merge_in_transpose(
                     noone_plan_src_to_dst,
                     noone_plan_dst_to_src,
                     noone_dims_src,
                     noone_dims_dst,
                     noone_rank);

         if (noone_merged_rank == 1) {
            if (parity) {
               for (Size k = 0; k < block_size; k++) {
                  dst_data[k] = -src_data[k];
               }
            } else {
               for (Size k = 0; k < block_size; k++) {
                  dst_data[k] = src_data[k];
               }
            }
         } else {
            Rank effective_rank = noone_merged_rank;
            Size effective_size = sizeof(ScalarType);
            if (noone_merged_plan_src_to_dst[noone_merged_rank - 1] == noone_merged_rank - 1) {
               effective_rank--;
               effective_size *= noone_merged_dims_dst[noone_merged_rank - 1];
            }
            // TODO: 需要考虑极端细致的情况
            if (parity) {
               block_transpose<true>(
                     src_data,
                     dst_data,
                     noone_merged_plan_src_to_dst,
                     noone_merged_plan_dst_to_src,
                     noone_merged_dims_src,
                     noone_merged_dims_dst,
                     block_size,
                     effective_rank,
                     effective_size);
            } else {
               block_transpose<false>(
                     src_data,
                     dst_data,
                     noone_merged_plan_src_to_dst,
                     noone_merged_plan_dst_to_src,
                     noone_merged_dims_src,
                     noone_merged_dims_dst,
                     block_size,
                     effective_rank,
                     effective_size);
            }
         }
      }

   private:
      Edge<Symmetry>
      get_single_merged_edge(const vector<EdgePointer<Symmetry>>& edges_to_merge) const {
         auto res_edge = Edge<Symmetry>();

         auto sym = vector<Symmetry>();
         auto dim = vector<Size>();

         using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;

         auto update_sym_and_dim = [&sym, &dim](const PosType& pos, Rank start) {
            for (Rank i = start; i < pos.size(); i++) {
               const auto& ptr = pos[i];
               if (i == 0) {
                  sym[i] = ptr->first;
                  dim[i] = ptr->second;
               } else {
                  sym[i] = ptr->first + sym[i - 1];
                  dim[i] = ptr->second * dim[i - 1];
                  // do not check dim=0, because in constructor, i didn't check
               }
            }
         };

         loop_edge(
               edges_to_merge,
               [&]() { res_edge[Symmetry()] = 1; },
               [&](const PosType& pos) {
                  sym.resize(pos.size());
                  dim.resize(pos.size());
                  update_sym_and_dim(pos, 0);
               },
               [&](const PosType& pos) {
                  auto index_to_check = Rank(pos.size());
                  for (const auto& block : core->blocks) {
                     for (Rank i = 0; i < index_to_check; i++) {
                        if (pos[i]->first != block.symmetries[i]) {
                           goto next_syms;
                        }
                     }
                     return true;
                  next_syms:;
                  }
                  return false;
               },
               [&]([[maybe_unused]] const PosType& pos) {
                  res_edge[sym[pos.size() - 1]] += dim[pos.size() - 1];
               },
               update_sym_and_dim);

         return res_edge;
      }

   public:
      struct NameWithEdge {
         Name name;
         Edge<Symmetry> edge;
      };
      // name:
      // 1. rename
      // 2. split
      // 3. merge
      // edge:
      // 1. split
      // 2. transpose
      // 3. merged
      template<class T = vector<Name>>
      Tensor<ScalarType, Symmetry> edge_operator(
            const std::map<Name, Name>& rename,
            const std::map<Name, vector<NameWithEdge>>& split,
            const std::map<Name, vector<Name>>& merge,
            T&& new_names,
            bool parity = false) const {
         // merge split的过程中, 会产生半个符号, 所以需要限定这一个tensor是否拥有这个符号
         vector<std::tuple<Rank, Rank>> split_list;
         vector<std::tuple<Rank, Rank>> merge_list;

         auto original_name = vector<Name>();
         auto splited_name = vector<Name>();
         const vector<Edge<Symmetry>>& original_edge = core->edges;
         auto splited_edge = vector<EdgePointer<Symmetry>>();
         for (auto& i : names) {
            Name renamed_name = i;
            auto it1 = rename.find(i);
            if (it1 != rename.end()) {
               renamed_name = it1->second;
            }
            original_name.push_back(renamed_name);

            const Edge<Symmetry>& e = core->edges[name_to_index.at(i)];
            auto it2 = split.find(renamed_name);
            if (it2 != split.end()) {
               auto edge_list = vector<EdgePointer<Symmetry>>();
               Rank split_start = splited_name.size();
               for (const auto& k : it2->second) {
                  splited_name.push_back(k.name);
                  splited_edge.push_back(&k.edge);
                  edge_list.push_back(&k.edge);
               }
               Rank split_end = splited_name.size();
               split_list.push_back({split_start, split_end});
               if (get_single_merged_edge(edge_list) != e) {
                  TAT_WARNING("Invalid Edge Split");
               }
            } else {
               splited_name.push_back(renamed_name);
               splited_edge.push_back(&e);
            }
         }

         const Rank original_rank = names.size();
         const Rank splited_rank = splited_name.size();

         auto splited_name_to_index = construct_name_to_index(splited_name);

         vector<Rank> fine_plan_dst;
         vector<Rank> plan_dst_to_src;

         vector<Name> merged_name = std::forward<T>(new_names);
         auto transposed_name = vector<Name>();
         auto merged_edge = vector<Edge<Symmetry>>();
         auto transposed_edge = vector<EdgePointer<Symmetry>>();

         Rank fine_dst_tmp = 0;
         for (const auto& i : merged_name) {
            auto it1 = merge.find(i);
            if (it1 != merge.end()) {
               auto edge_list = vector<EdgePointer<Symmetry>>();
               Rank merge_start = transposed_name.size();
               for (const auto& k : it1->second) {
                  transposed_name.push_back(k);
                  auto idx = splited_name_to_index.at(k);
                  plan_dst_to_src.push_back(idx);
                  fine_plan_dst.push_back(fine_dst_tmp);
                  auto ep = splited_edge[idx];
                  transposed_edge.push_back(ep);
                  edge_list.push_back(ep);
               }
               Rank merge_end = transposed_name.size();
               merge_list.push_back({merge_start, merge_end});
               merged_edge.push_back(get_single_merged_edge(edge_list));
            } else {
               transposed_name.push_back(i);
               auto idx = splited_name_to_index.at(i);
               plan_dst_to_src.push_back(idx);
               fine_plan_dst.push_back(fine_dst_tmp);
               auto ep = splited_edge[idx];
               transposed_edge.push_back(ep);
               merged_edge.push_back(*ep.ptr);
            }
            fine_dst_tmp++;
         }

         const Rank transposed_rank = transposed_name.size();
         const Rank merged_rank = merged_name.size();

         vector<Rank> fine_plan_src(transposed_rank);
         vector<Rank> plan_src_to_dst(transposed_rank);

         for (Rank i = 0; i < transposed_rank; i++) {
            plan_src_to_dst[plan_dst_to_src[i]] = i;
         }

         Rank fine_src_tmp = 0;
         for (const auto& i : original_name) {
            auto it1 = split.find(i);
            if (it1 != split.end()) {
               for (const auto& j : it1->second) {
                  fine_plan_src[plan_src_to_dst[splited_name_to_index.at(j.name)]] = fine_src_tmp;
               }
            } else {
               fine_plan_src[plan_src_to_dst[splited_name_to_index.at(i)]] = fine_src_tmp;
            }
            fine_src_tmp++;
         }

         if (merged_rank == original_rank) {
            for (Rank i = 0; i < merged_rank; i++) {
               Name name1 = original_name[i];
               auto it1 = split.find(name1);
               if (it1 != split.end()) {
                  if (it1->second.size() != 1) {
                     goto do_operator;
                  }
                  name1 = it1->second[0].name;
               }
               Name name2 = merged_name[i];
               auto it2 = merge.find(name2);
               if (it2 != merge.end()) {
                  if (it2->second.size() != 1) {
                     goto do_operator;
                  }
                  name2 = it2->second[0];
               }
               if (name1 != name2) {
                  goto do_operator;
               }
            }
            auto res = Tensor<ScalarType, Symmetry>{};
            res.names = std::move(merged_name);
            res.name_to_index = construct_name_to_index(res.names);
            res.core = core;
            return res;
         }
      do_operator:
         auto res = Tensor<ScalarType, Symmetry>(std::move(merged_name), std::move(merged_edge));

         Rank src_rank = names.size();
         Rank dst_rank = res.names.size();
         Nums src_block_number = Nums(core->blocks.size());
         Nums dst_block_number = Nums(res.core->blocks.size());
         vector<Size> src_offset(src_block_number, 0);
         vector<Size> dst_offset(dst_block_number, 0);

         using PosType = vector<typename Edge<Symmetry>::const_iterator>;

         loop_edge(
               transposed_edge,
               // rank0
               []() {
                  // TODO: rank0
                  // only one element, need to check parity
               },
               // init
               []([[maybe_unused]] const PosType& pos) {},
               // check
               []([[maybe_unused]] const PosType& pos) {
                  auto sum = Symmetry();
                  for (const auto& i : pos) {
                     sum += i->first;
                  }
                  return sum == Symmetry();
               },
               // append
               [&]([[maybe_unused]] const PosType& pos) {
                  vector<Symmetry> src_pos(src_rank, Symmetry());
                  vector<Symmetry> dst_pos(dst_rank, Symmetry());
                  for (Rank i = 0; i < transposed_rank; i++) {
                     src_pos[fine_plan_src[i]] += pos[i]->first;
                     dst_pos[fine_plan_dst[i]] += pos[i]->first;
                  }
                  auto src_block_index = core->find_block(src_pos);
                  auto dst_block_index = res.core->find_block(dst_pos);
                  auto src_data =
                        core->blocks[src_block_index].raw_data.data() + src_offset[src_block_index];
                  auto dst_data = res.core->blocks[dst_block_index].raw_data.data() +
                                  src_offset[dst_block_index];

                  vector<Size> dst_dim(transposed_rank);
                  vector<Size> src_dim(transposed_rank);
                  for (Rank i = 0; i < transposed_rank; i++) {
                     src_dim[plan_dst_to_src[i]] = dst_dim[i] = pos[i]->second;
                  }

                  Size total_size = 1;
                  for (auto i : dst_dim) {
                     total_size *= i;
                  }

                  bool p = Symmetry::get_parity(src_pos, plan_src_to_dst);
                  if (parity) {
                     p ^= Symmetry::get_parity(src_pos, split_list) ^
                          Symmetry::get_parity(dst_pos, merge_list);
                  }
                  do_transpose(
                        plan_src_to_dst,
                        plan_dst_to_src,
                        src_data,
                        dst_data,
                        src_dim,
                        dst_dim,
                        total_size,
                        transposed_rank,
                        p);
                  // TODO: offset这里似乎不对
                  // src_offset[src_block_index] += total_size;
                  // dst_offset[dst_block_index] += total_size;
               },
               // update
               // TODO: 一些append里的东西可以放在update里
               []([[maybe_unused]] const PosType& pos, [[maybe_unused]] Rank ptr) {});

         return res;
      }
      // TODO: loop_edge调用的时候使用as_const, 明确引用

   public:
      // 过早的优化时一切罪恶的根源！！！！

      // TODO: merge and split
      // 原则上可以一边转置一边 merge split
      // merge 的时候会产生半个parity符号
      // 所以这个接口需要有个bool来确定是否处理符号
      // contract 调用的时候， 两个张量一个带符号一个不带即可

      // TODO: contract
      // 调用 merge ， 这样就不用考虑contract特有的符号问题了
      static Tensor<ScalarType, Symmetry> contract(
            const Tensor<ScalarType, Symmetry>& tensor1,
            const Tensor<ScalarType, Symmetry>& tensor2,
            const vector<Name>& names1,
            const vector<Name>& names2) {
         // TODO: 不转置成矩阵直接乘积的可能
         // names1 names2 需要 check order, 这个无法再merge中判断
         // merge 需要按照原样的顺序进行转置
         auto merged_tensor1 = tensor1.merge_edge({{names1, Contract1}}, false, true);
         auto merged_tensor2 = tensor2.merge_edge({{names2, Contract2}}, true, true);
         // check which one in 8 and do matrix contract
      }

      // TODO: converge
      // multiple 的扩展版
      // 考虑到PESS可能和对称性不兼容, 那么还是恢复原来的multiple模式
      // 因为这个东西唯一出现的地方就是svd了
      static Tensor<ScalarType, Symmetry> converge(
            const Tensor<ScalarType, Symmetry>& tensor1,
            const Tensor<ScalarType, Symmetry>& tensor2,
            const vector<Name>& names1,
            const vector<Name>& names2,
            const vector<Name>& names_res,
            const Size order) {
         // 这个东西在svd写好后再弄
      }

      struct svd_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<real_base_t<ScalarType>, Symmetry> S;
         Tensor<ScalarType, Symmetry> V;
      };

      // TODO: SVD
      // 根据情况选择转置方式
      // 或许可以称作cutted orthogonalize
      // 需要先merge再split, 不然不能保证分块依然分块
      svd_res svd(const vector<Name>& u_edges, Name u_new_name, Name v_new_name) const {
         // auto merged_tensor = merge_edge({{u_edges, SVD1}, {v_edges, SVD2}});
         // svd
         // auto split...
      }

      struct orthogonalize_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<ScalarType, Symmetry> T;
      };

      // TODO: QR LQ
      // 除了svd的考虑外， 需要考虑使用lq还是qr等
      orthogonalize_res orthogonalize(
            const vector<Name>& given_edges,
            Name given_new_name,
            Name other_new_name,
            bool u_edges_given = true) const {
         //
      }

      // TODO: 代码review
      // TODO: 更美观的IO

      const Tensor<ScalarType, Symmetry>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry>& data_get(std::istream&);
   }; // namespace TAT

#   define DEF_SCALAR_OP(OP, EVAL1, EVAL2, EVAL3)                                                 \
      template<class ScalarType1, class ScalarType2, class Symmetry>                              \
      auto OP(const Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) { \
         using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;                         \
         if (t1.names.size() == 0) {                                                              \
            const auto& x = t1.core->blocks[0].raw_data[0];                                       \
            auto res = Tensor<ScalarType, Symmetry>{t2.names, t2.core->edges};                    \
            auto blocks_number = Nums(res.core->blocks.size());                                   \
            for (Nums i = 0; i < blocks_number; i++) {                                            \
               const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();              \
               ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
               Size block_size = res.core->blocks[i].size;                                        \
               for (Size j = 0; j < block_size; j++) {                                            \
                  EVAL1;                                                                          \
               }                                                                                  \
            }                                                                                     \
            return res;                                                                           \
         } else if (t2.names.size() == 0) {                                                       \
            const auto& y = t2.core->blocks[0].raw_data[0];                                       \
            auto res = Tensor<ScalarType, Symmetry>{t1.names, t1.core->edges};                    \
            auto blocks_number = Nums(res.core->blocks.size());                                   \
            for (Nums i = 0; i < blocks_number; i++) {                                            \
               const ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();              \
               ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
               Size block_size = res.core->blocks[i].size;                                        \
               for (Size j = 0; j < block_size; j++) {                                            \
                  EVAL2;                                                                          \
               }                                                                                  \
            }                                                                                     \
            return res;                                                                           \
         } else {                                                                                 \
            if (!((t1.names == t2.names) && (t1.core->edges == t2.core->edges))) {                \
               TAT_WARNING("Scalar Operator In Different Shape Tensor");                          \
            }                                                                                     \
            auto res = Tensor<ScalarType, Symmetry>{t1.names, t1.core->edges};                    \
            auto blocks_number = Nums(res.core->blocks.size());                                   \
            for (Nums i = 0; i < blocks_number; i++) {                                            \
               const ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();              \
               const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();              \
               ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
               Size block_size = res.core->blocks[i].size;                                        \
               for (Size j = 0; j < block_size; j++) {                                            \
                  EVAL3;                                                                          \
               }                                                                                  \
            }                                                                                     \
            return res;                                                                           \
         }                                                                                        \
      }                                                                                           \
      template<                                                                                   \
            class ScalarType1,                                                                    \
            class ScalarType2,                                                                    \
            class Symmetry,                                                                       \
            class = std::enable_if_t<is_scalar_v<ScalarType2>>>                                   \
      auto OP(const Tensor<ScalarType1, Symmetry>& t1, const ScalarType2& n2) {                   \
         return OP(t1, Tensor<ScalarType2, Symmetry>{n2});                                        \
      }                                                                                           \
      template<                                                                                   \
            class ScalarType1,                                                                    \
            class ScalarType2,                                                                    \
            class Symmetry,                                                                       \
            class = std::enable_if_t<is_scalar_v<ScalarType1>>>                                   \
      auto OP(const ScalarType1& n1, const Tensor<ScalarType2, Symmetry>& t2) {                   \
         return OP(Tensor<ScalarType1, Symmetry>{n1}, t2);                                        \
      }

   DEF_SCALAR_OP(operator+, c[j] = x + b[j], c[j] = a[j] + y, c[j] = a[j] + b[j])
   DEF_SCALAR_OP(operator-, c[j] = x - b[j], c[j] = a[j] - y, c[j] = a[j] - b[j])
   DEF_SCALAR_OP(operator*, c[j] = x* b[j], c[j] = a[j]* y, c[j] = a[j]* b[j])
   DEF_SCALAR_OP(operator/, c[j] = x / b[j], c[j] = a[j] / y, c[j] = a[j] / b[j])
#   undef DEF_SCALAR_OP

#   define DEF_SCALAR_OP(OP, EVAL1, EVAL2)                                                \
      template<class ScalarType1, class ScalarType2, class Symmetry>                      \
      Tensor<ScalarType1, Symmetry>& OP(                                                  \
            Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) { \
         if (t1.core.use_count() != 1) {                                                  \
            TAT_WARNING("Inplace Operator On Tensor Shared");                             \
         }                                                                                \
         if (t2.names.size() == 0) {                                                      \
            const auto& y = t2.core->blocks[0].raw_data[0];                               \
            Nums blocks_number = t1.core->blocks.size();                                  \
            for (Nums i = 0; i < blocks_number; i++) {                                    \
               ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();            \
               Size block_size = t1.core->blocks[i].size;                                 \
               for (Size j = 0; j < block_size; j++) {                                    \
                  EVAL1;                                                                  \
               }                                                                          \
            }                                                                             \
         } else {                                                                         \
            if (!((t1.names == t2.names) && (t1.core->edges == t2.core->edges))) {        \
               TAT_WARNING("Scalar Operator In Different Shape Tensor");                  \
            }                                                                             \
            Nums blocks_number = t1.core->blocks.size();                                  \
            for (Nums i = 0; i < blocks_number; i++) {                                    \
               ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();            \
               const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();      \
               Size block_size = t1.core->blocks[i].size;                                 \
               for (Size j = 0; j < block_size; j++) {                                    \
                  EVAL2;                                                                  \
               }                                                                          \
            }                                                                             \
         }                                                                                \
         return t1;                                                                       \
      }                                                                                   \
      template<                                                                           \
            class ScalarType1,                                                            \
            class ScalarType2,                                                            \
            class Symmetry,                                                               \
            class = std::enable_if_t<is_scalar_v<ScalarType2>>>                           \
      Tensor<ScalarType1, Symmetry>& OP(                                                  \
            Tensor<ScalarType1, Symmetry>& t1, const ScalarType2& n2) {                   \
         return OP(t1, Tensor<ScalarType2, Symmetry>{n2});                                \
      }
   DEF_SCALAR_OP(operator+=, a[j] += y, a[j] += b[j])
   DEF_SCALAR_OP(operator-=, a[j] -= y, a[j] -= b[j])
   DEF_SCALAR_OP(operator*=, a[j] *= y, a[j] *= b[j])
   DEF_SCALAR_OP(operator/=, a[j] /= y, a[j] /= b[j])
#   undef DEF_SCALAR_OP

   // TODO: lazy framework
   // 需要考虑深搜不可行的问题
   // 支持inplace操作

   // GPU and so on
} // namespace TAT

#endif
