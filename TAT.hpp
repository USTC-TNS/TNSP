/**
 * \file TAT.hpp
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

#ifndef TAT_HPP_
#define TAT_HPP_

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#ifdef NDEBUG
#   define TAT_WARNING(msg) std::clog
#else
#   define TAT_WARNING(msg) std::clog << msg << std::endl
#endif

namespace TAT {
   using Rank = unsigned short;
   using Nums = unsigned int;
   using Size = unsigned long;
   using Z2 = bool;
   using U1 = long;
   using Fermi = int;

   template<class T>
   struct is_scalar : std::is_scalar<T> {};
   template<class T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<class T>
   static constexpr bool is_scalar_v = is_scalar<T>::value;

   template<class T>
   struct type_identity {
      using type = T;
   };
   template<class T>
   using type_identity_t = typename type_identity<T>::type;

   template<class T>
   struct real_base : type_identity<T> {};
   template<class T>
   struct real_base<std::complex<T>> : type_identity<T> {};
   template<class T>
   using real_base_t = typename real_base<T>::type;

   template<class T>
   struct remove_cvref : std::remove_cv<typename std::remove_reference<T>::type> {};
   template<class T>
   using remove_cvref_t = typename remove_cvref<T>::type;

   template<class T, class U>
   struct is_same_nocvref : std::is_same<remove_cvref_t<T>, remove_cvref_t<U>> {};
   template<class T, class U>
   static constexpr bool is_same_nocvref_v = is_same_nocvref<T, U>::value;

   using NameIdType = int;

   NameIdType names_total = 0;
   std::map<std::string, NameIdType> name_to_id = {};
   std::map<NameIdType, std::string> id_to_name = {};

   struct Name {
      NameIdType id = -1;
      Name() = default;
      Name(NameIdType id) : id{id} {}
      Name(const std::string& name) {
         auto pos = name_to_id.find(name);
         if (pos == name_to_id.end()) {
            id = names_total++;
            name_to_id[name] = id;
            id_to_name[id] = name;
         } else {
            id = pos->second;
         }
      }
   };

   std::ostream& operator<<(std::ostream& out, const Name& name);
   std::istream& operator>>(std::istream& in, Name& name);

#define TAT_DEF_NAME_OP(OP, EXP)           \
   bool OP(const Name& a, const Name& b) { \
      return EXP;                          \
   }
   TAT_DEF_NAME_OP(operator==, a.id == b.id);
   TAT_DEF_NAME_OP(operator!=, a.id != b.id);
   TAT_DEF_NAME_OP(operator>=, a.id >= b.id);
   TAT_DEF_NAME_OP(operator<=, a.id <= b.id);
   TAT_DEF_NAME_OP(operator>, a.id> b.id);
   TAT_DEF_NAME_OP(operator<, a.id<b.id);
#undef TAT_DEF_NAME_OP

#define TAT_DEF_NAME(x) const Name x(#x)
#define TAT_DEF_NAMES(n)      \
   TAT_DEF_NAME(Phy##n);      \
   TAT_DEF_NAME(Left##n);     \
   TAT_DEF_NAME(Right##n);    \
   TAT_DEF_NAME(Up##n);       \
   TAT_DEF_NAME(Down##n);     \
   TAT_DEF_NAME(LeftUp##n);   \
   TAT_DEF_NAME(LeftDown##n); \
   TAT_DEF_NAME(RightUp##n);  \
   TAT_DEF_NAME(RightDown##n)
   TAT_DEF_NAMES();
   TAT_DEF_NAMES(1);
   TAT_DEF_NAMES(2);
   TAT_DEF_NAMES(3);
   TAT_DEF_NAMES(4);
#undef TAT_DEF_NAMES
#undef TAT_DEF_NAME

   template<class T>
   struct allocator_without_initialize : std::allocator<T> {
      template<class U>
      struct rebind {
         using other = allocator_without_initialize<U>;
      };

      template<class... Args>
      void construct([[maybe_unused]] T* p, Args&&... args) {
         if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
            new (p) T(args...);
         }
      }

      allocator_without_initialize() = default;
      template<class U>
      allocator_without_initialize(allocator_without_initialize<U>) {}
   };

   template<class T>
   struct vector : public std::vector<T, allocator_without_initialize<T>> {
      using std::vector<T, allocator_without_initialize<T>>::vector;
   };

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec);
   template<class T>
   std::istream& operator>>(std::istream& in, vector<T>& vec);

   template<class Symmetry>
   struct Edge : public std::map<Symmetry, Size> {
      using std::map<Symmetry, Size>::map;

      Edge(Size s) : std::map<Symmetry, Size>({{Symmetry(), s}}) {}
   };

   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge);

   template<class Symmetry>
   std::istream& operator>>(std::istream& in, Edge<Symmetry>& edge);

   template<class ScalarType, class Symmetry>
   struct Block {
      const vector<Edge<Symmetry>>& edges;
      vector<Symmetry> symmetries;
      vector<ScalarType> raw_data;
      Size size;

      template<
            class T = vector<Symmetry>,
            class = std::enable_if_t<is_same_nocvref_v<T, vector<Symmetry>>>>
      Block(const vector<Edge<Symmetry>>& e, T&& s) : edges(e), symmetries(std::forward<T>(s)) {
         size = 1;
         for (Rank i = 0; i < edges.size(); i++) {
            size *= edges[i].at(symmetries[i]);
         }
         raw_data = vector<ScalarType>(size);
      }
   };

   struct NoSymmetry {
      static bool check_symmetry_satisfied(const vector<NoSymmetry>&) {
         return true;
      }
   };

   std::ostream& operator<<(std::ostream& out, const NoSymmetry&);
   std::istream& operator>>(std::istream& in, NoSymmetry&);
#define TAT_DEF_SYM_OP(OP, EXP)                    \
   bool OP(const NoSymmetry&, const NoSymmetry&) { \
      return EXP;                                  \
   }
   TAT_DEF_SYM_OP(operator==, true);
   TAT_DEF_SYM_OP(operator!=, false);
   TAT_DEF_SYM_OP(operator>=, true);
   TAT_DEF_SYM_OP(operator<=, true);
   TAT_DEF_SYM_OP(operator>, false);
   TAT_DEF_SYM_OP(operator<, false);
#undef TAT_DEF_SYM_OP

   struct Z2Symmetry {
      Z2 z2 = 0;

      Z2Symmetry(Z2 z2 = 0) : z2(z2) {}

      static bool check_symmetry_satisfied(const vector<Z2Symmetry>& vec) {
         Z2 sum = 0;
         for (const auto& i : vec) {
            sum ^= i.z2;
         }
         return !sum;
      };
   };
   std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s);
   std::istream& operator>>(std::istream& in, Z2Symmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                        \
   bool OP(const Z2Symmetry& a, const Z2Symmetry& b) { \
      return EXP;                                      \
   }
   TAT_DEF_SYM_OP(operator==, a.z2 == b.z2);
   TAT_DEF_SYM_OP(operator!=, a.z2 != b.z2);
   TAT_DEF_SYM_OP(operator>=, a.z2 >= b.z2);
   TAT_DEF_SYM_OP(operator<=, a.z2 <= b.z2);
   TAT_DEF_SYM_OP(operator>, a.z2> b.z2);
   TAT_DEF_SYM_OP(operator<, a.z2<b.z2);
#undef TAT_DEF_SYM_OP

   struct U1Symmetry {
      U1 u1 = 0;

      U1Symmetry(U1 u1 = 0) : u1(u1) {}

      static bool check_symmetry_satisfied(const vector<U1Symmetry>& vec) {
         U1 sum = 0;
         for (const auto& i : vec) {
            sum += i.u1;
         }
         return !sum;
      }
   };
   std::ostream& operator<<(std::ostream& out, const U1Symmetry& s);
   std::istream& operator>>(std::istream& in, U1Symmetry& s);

#define TAT_DEF_SYM_OP(OP, EXP)                        \
   bool OP(const U1Symmetry& a, const U1Symmetry& b) { \
      return EXP;                                      \
   }
   TAT_DEF_SYM_OP(operator==, a.u1 == b.u1);
   TAT_DEF_SYM_OP(operator!=, a.u1 != b.u1);
   TAT_DEF_SYM_OP(operator>=, a.u1 >= b.u1);
   TAT_DEF_SYM_OP(operator<=, a.u1 <= b.u1);
   TAT_DEF_SYM_OP(operator>, a.u1> b.u1);
   TAT_DEF_SYM_OP(operator<, a.u1<b.u1);
#undef TAT_DEF_SYM_OP

   struct FermiSymmetry {
      Fermi fermi = 0;
   };

   struct FermiZ2Symmetry {
      Fermi fermi = 0;
      Z2 z2 = 0;
   };

   struct FermiU1Symmetry {
      Fermi fermi = 0;
      U1 u1 = 0;
   };

   template<class Symmetry>
   auto initialize_block_symmetries_with_check(const vector<Edge<Symmetry>>& edges) {
      auto rank = edges.size();
      if (!rank) {
         return vector<vector<Symmetry>>{{}};
      }
      auto pos = vector<typename std::map<Symmetry, Size>::const_iterator>();
      auto vec = vector<Symmetry>();
      auto res = vector<vector<Symmetry>>();
      for (const auto& i : edges) {
         auto ptr = i.begin();
         if (ptr == i.end()) {
            return res;
         }
         pos.push_back(ptr);
         vec.push_back(ptr->first);
      }
      while (true) {
         if (Symmetry::check_symmetry_satisfied(vec)) {
            res.push_back(vec);
         }
         Rank ptr = 0;
         pos[ptr]++;
         while (pos[ptr] == edges[ptr].end()) {
            pos[ptr] = edges[ptr].begin();
            vec[ptr] = pos[ptr]->first;
            ptr++;
            if (ptr == rank) {
               return res;
            }
            pos[ptr]++;
         }
         vec[ptr] = pos[ptr]->first;
      }
   }

   std::map<Name, Rank> construct_name_to_index(const vector<Name>& names) {
      std::map<Name, Rank> res;
      Rank rank = names.size();
      for (Rank i = 0; i < rank; i++) {
         res[names[i]] = i;
      }
      return res;
   }

   // TODO: transpose的优化
   // stupid_transpose
   // 最后一维相同
   // -- copy_transpose
   // -- 如果最后一列太小, 需要考虑把一列作为一个数据类型然后相同的手段, block_copy_transpose
   // 最后一维不同
   // -- block_transpose 调用matrix_transpose
   // -- 如果两个最后维之一的太小， 需要考虑小维度的上面一维， 如果都小， 都要考虑
   // 注意：可以fuse的先fuse
   // 注意: 如果不需要转置, 则返回shared_ptr相同的core
   template<class ScalarType>
   void stupid_matrix_transpose(
         Size M,
         Size N,
         const ScalarType* __restrict src,
         Size leading_src,
         ScalarType* __restrict dst,
         Size leading_dst) {
      for (Size i = 0; i < M; i++) {
         for (Size j = 0; j < N; j++) {
            dst[j * leading_dst + i] = src[i * leading_src + j];
         }
      }
   }

   template<class ScalarType>
   void stupid_transpose(
         const ScalarType* __restrict src,
         ScalarType* __restrict dst,
         [[maybe_unused]] const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         [[maybe_unused]] const Size& size,
         const Rank& rank) {
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

      while (1) {
         dst[index_dst] = src[index_src];

         Rank temp_rank_dst = rank - 1;
         Rank temp_rank_src = plan_dst_to_src[temp_rank_dst];

         index_list_src[temp_rank_src] += 1;
         index_list_dst[temp_rank_dst] += 1;
         index_src += step_src[temp_rank_src];
         index_dst += step_dst[temp_rank_dst];

         while (index_list_dst[temp_rank_dst] == dims_dst[temp_rank_dst]) {
            if (temp_rank_dst == 0) {
               return;
            }
            index_list_src[temp_rank_src] = 0;
            index_src -= dims_src[temp_rank_src] * step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] = 0;
            index_dst -= dims_dst[temp_rank_dst] * step_dst[temp_rank_dst];
            temp_rank_dst -= 1;
            temp_rank_src = plan_dst_to_src[temp_rank_dst];
            index_list_src[temp_rank_src] += 1;
            index_src += step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] += 1;
            index_dst += step_dst[temp_rank_dst];
         }
      }
   }

   template<class ScalarType>
   void copy_transpose(
         const ScalarType* __restrict src,
         ScalarType* __restrict dst,
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         const Size& size,
         const Rank& rank) {
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

      if (rank == 1) {
         for (Size i = 0; i < size; i++) {
            dst[i] = src[i];
         }
         return;
      }
      Size last_dim = dims_dst[rank - 1];
      while (1) {
         for (Size i = 0; i < last_dim; i++) {
            dst[index_dst + i] = src[index_src + i];
         }

         Rank temp_rank_dst = rank - 2;
         Rank temp_rank_src = plan_dst_to_src[temp_rank_dst];

         index_list_src[temp_rank_src] += 1;
         index_src += step_src[temp_rank_src];
         index_list_dst[temp_rank_dst] += 1;
         index_dst += step_dst[temp_rank_dst];

         while (index_list_dst[temp_rank_dst] == dims_dst[temp_rank_dst]) {
            if (temp_rank_dst == 0) {
               return;
            }
            index_list_src[temp_rank_src] = 0;
            index_src -= dims_src[temp_rank_src] * step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] = 0;
            index_dst -= dims_dst[temp_rank_dst] * step_dst[temp_rank_dst];
            temp_rank_dst -= 1;
            temp_rank_src = plan_dst_to_src[temp_rank_dst];
            index_list_src[temp_rank_src] += 1;
            index_src += step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] += 1;
            index_dst += step_dst[temp_rank_dst];
         }
      }
   }

   template<class ScalarType>
   void block_transpose(
         const ScalarType* __restrict src,
         ScalarType* __restrict dst,
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         const Size& size,
         const Rank& rank) {
      stupid_transpose(src, dst, plan_src_to_dst, plan_dst_to_src, dims_src, dims_dst, size, rank);
   }

   template<class Symmetry>
   bool check_same_symmetries(
         const vector<Symmetry>& s1,
         const vector<Symmetry>& s2,
         const vector<Rank>& plan) {
      Rank rank = plan.size();
      for (Rank i = 0; i < rank; i++) {
         if (s1[i] != s2[plan[i]]) {
            return false;
         }
      }
      return true;
   }

   template<class ScalarType = double, class Symmetry = NoSymmetry>
   struct Tensor {
      struct TensorCore {
         vector<Edge<Symmetry>> edges;
         vector<Block<ScalarType, Symmetry>> blocks;

         template<
               class T = vector<Edge<Symmetry>>,
               class = std::enable_if_t<is_same_nocvref_v<T, vector<Edge<Symmetry>>>>>
         TensorCore(T&& e) : edges(std::forward<T>(e)) {
            auto symmetries_list = initialize_block_symmetries_with_check<Symmetry>(edges);
            for (auto& i : symmetries_list) {
               blocks.push_back(Block<ScalarType, Symmetry>(edges, std::move(i)));
            }
         }
      };

      vector<Name> names;
      std::map<Name, Rank> name_to_index;
      std::shared_ptr<TensorCore> core;

      bool is_valid_name() const {
         return names.size() == std::set<Name>(names.begin(), names.end()).size() &&
                names.size() == core->edges.size();
      }

      template<
            class U = vector<Name>,
            class T = vector<Edge<Symmetry>>,
            class = std::enable_if_t<is_same_nocvref_v<U, vector<Name>>>,
            class = std::enable_if_t<is_same_nocvref_v<T, vector<Edge<Symmetry>>>>>
      Tensor(U&& n, T&& e) :
            names(std::forward<U>(n)), name_to_index(construct_name_to_index(names)),
            core(std::make_shared<TensorCore>(std::forward<T>(e))) {
         if (!is_valid_name()) {
            TAT_WARNING("Invalid Names");
         }
      }

      Tensor() = default;
      Tensor(const Tensor& other) :
            names(other.names), name_to_index(other.name_to_index),
            core(std::make_shared<TensorCore>(*other.core)) {
         TAT_WARNING("Data Copy in Tensor Copy");
      }
      Tensor(Tensor&& other) = default;
      ~Tensor() = default;
      Tensor& operator=(const Tensor& other) {
         names = other.names;
         name_to_index = other.name_to_index;
         core = std::make_shared<TensorCore>(*other.core);
         TAT_WARNING("Data Copy in Tensor Copy");
      }
      Tensor& operator=(Tensor&& other) = default;

      Tensor(ScalarType num) : Tensor({}, {}) {
         core->blocks[0].raw_data[0] = num;
      }

      operator ScalarType() const {
         if (names.size() != 0) {
            TAT_WARNING("Conversion From multiple rank Tensor to Scalar");
         }
         return core->blocks[0].raw_data[0];
      }

      template<class Generator>
      Tensor& set(Generator&& generator) & {
         if (core.use_count() != 1) {
            TAT_WARNING("Set Tensor Shared");
         }
         for (auto& i : core->blocks) {
            std::generate(i.raw_data.begin(), i.raw_data.begin() + i.size, generator);
         }
         return *this;
      }
      template<class Generator>
      Tensor<ScalarType, Symmetry>&& set(Generator&& generator) && {
         return std::move(set(generator));
      }

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
            res.core = std::make_shared<typename Tensor<OtherScalarType, Symmetry>::TensorCore>(
                  core->edges);
            Nums blocks_num = core->blocks.size();
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
      };

      template<int p = 2>
      Tensor<real_base_t<ScalarType>, Symmetry> norm() const {
         real_base_t<ScalarType> res = 0;
         if constexpr (p == -1) {
            Nums blocks_num = core->blocks.size();
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
            Nums blocks_num = core->blocks.size();
            for (Nums i = 0; i < blocks_num; i++) {
               res += core->blocks[i].size;
            }
         } else {
            Nums blocks_num = core->blocks.size();
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

      Tensor<ScalarType, Symmetry> edge_rename(const std::map<Name, Name>& dict) const {
         auto res = Tensor<ScalarType, Symmetry>{};
         res.core = core;
         std::transform(names.begin(), names.end(), std::back_inserter(res.names), [&](Name name) {
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

      template<class T = vector<Name>, class = std::enable_if_t<is_same_nocvref_v<T, vector<Name>>>>
      Tensor<ScalarType, Symmetry> transpose(T&& target_names) const {
         auto res = Tensor<ScalarType, Symmetry>{};
         res.names = std::forward<T>(target_names);
         res.name_to_index = construct_name_to_index(res.names);

         Rank rank = names.size();
         vector<Rank> plan_src_to_dst(rank);
         vector<Rank> plan_dst_to_src(rank);
         vector<Edge<Symmetry>> res_edges(rank);
         for (Rank i = 0; i < rank; i++) {
            plan_src_to_dst[i] = res.name_to_index.at(names[i]);
            plan_dst_to_src[plan_src_to_dst[i]] = i;
            res_edges[plan_src_to_dst[i]] = core->edges[i];
         }
         res.core = std::make_shared<TensorCore>(std::move(res_edges));

         Nums block_number = core->blocks.size();
         for (Nums index_src = 0; index_src < block_number; index_src++) {
            Nums index_dst = 0;
            while (!check_same_symmetries(
                  core->blocks[index_src].symmetries,
                  res.core->blocks[index_dst].symmetries,
                  plan_src_to_dst)) {
               index_dst++;
            }
            Size block_size = core->blocks[index_src].size;
            if (block_size == 1) {
               res.core->blocks[index_dst].raw_data[0] = core->blocks[index_src].raw_data[0];
            } else {
               vector<Size> dims_src(rank);
               vector<Size> dims_dst(rank);
               for (Rank i = 0; i < rank; i++) {
                  dims_src[i] = core->edges[i].at(core->blocks[index_src].symmetries[i]);
                  dims_dst[plan_src_to_dst[i]] = dims_src[i];
               }
               if (plan_src_to_dst[rank - 1] == rank - 1) {
                  // if (dims_src[rank-1] < 4) {
                  //   big_block_transpose();
                  // } else
                  copy_transpose<ScalarType>(
                        core->blocks[index_src].raw_data.data(),
                        res.core->blocks[index_dst].raw_data.data(),
                        plan_src_to_dst,
                        plan_dst_to_src,
                        dims_src,
                        dims_dst,
                        block_size,
                        rank);
               } else {
                  // if (dims_src[rank-1] < 4 && dims_dst[rank-1] < 4) then ... difficult
                  // else if (dims_src[rank-1] < 4) then 3 way transpose
                  // else if (dims_dst[rank-1] < 4) then 3 way transpose
                  // else
                  block_transpose<ScalarType>(
                        core->blocks[index_src].raw_data.data(),
                        res.core->blocks[index_dst].raw_data.data(),
                        plan_src_to_dst,
                        plan_dst_to_src,
                        dims_src,
                        dims_dst,
                        block_size,
                        rank);
               }
            }
         }
         return res;
      }

      // TODO: contract
      // 包括了multiple
      static Tensor<ScalarType, Symmetry> contract(
            const Tensor<ScalarType, Symmetry>& t1,
            const Tensor<ScalarType, Symmetry>& t2,
            const vector<Name>& n1,
            const vector<Name>& n2,
            const vector<Name>& common = {}) {
         // 先确定一个基调就是先转置成矩阵
         // 而不是直接做张量乘积，不然太复杂了
         // A*B -> C
         // 有2^3=8种转置情况，需要考虑哪种方式快捷，从而使用
         // 所以还是需要把转置弄清楚啊， 不然都不知道如何转置快如何慢
      }

      struct svd_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<real_base_t<ScalarType>, Symmetry> S;
         Tensor<ScalarType, Symmetry> V;
      };

      // TODO: SVD
      // 根据情况选择转置方式
      svd_res svd(const vector<Name>& u_edges, Name u_new_name, Name v_new_name) const {
         //
      }

      struct orthogonalize_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<ScalarType, Symmetry> T;
      };

      // TODO: qr lq
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
   };

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor);
   template<class ScalarType, class Symmetry>
   std::istream& operator>>(std::istream& in, Tensor<ScalarType, Symmetry>& tensor);

#define DEF_SCALAR_OP(OP, EVAL1, EVAL2, EVAL3)                                                 \
   template<class ScalarType1, class ScalarType2, class Symmetry>                              \
   auto OP(const Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) { \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;                         \
      if (t1.names.size() == 0) {                                                              \
         const auto& x = t1.core->blocks[0].raw_data[0];                                       \
         auto res = Tensor<ScalarType, Symmetry>{t2.names, t2.core->edges};                    \
         Nums blocks_number = res.core->blocks.size();                                         \
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
         Nums blocks_number = res.core->blocks.size();                                         \
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
            TAT_WARNING("Scalar Operator in different Shape Tensor");                          \
         }                                                                                     \
         auto res = Tensor<ScalarType, Symmetry>{t1.names, t1.core->edges};                    \
         Nums blocks_number = res.core->blocks.size();                                         \
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
#undef DEF_SCALAR_OP

#define DEF_SCALAR_OP(OP, EVAL1, EVAL2)                                                          \
   template<class ScalarType1, class ScalarType2, class Symmetry>                                \
   Tensor<ScalarType1, Symmetry>& OP(                                                            \
         Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) {           \
      if (t1.core.use_count() != 1) {                                                            \
         TAT_WARNING("Inplace Operator on Tensor Shared");                                       \
      }                                                                                          \
      if (t2.names.size() == 0) {                                                                \
         const auto& y = t2.core->blocks[0].raw_data[0];                                         \
         Nums blocks_number = t1.core->blocks.size();                                            \
         for (Nums i = 0; i < blocks_number; i++) {                                              \
            ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();                      \
            Size block_size = t1.core->blocks[i].size;                                           \
            for (Size j = 0; j < block_size; j++) {                                              \
               EVAL1;                                                                            \
            }                                                                                    \
         }                                                                                       \
      } else {                                                                                   \
         if (!((t1.names == t2.names) && (t1.core->edges == t2.core->edges))) {                  \
            TAT_WARNING("Scalar Operator in different Shape Tensor");                            \
         }                                                                                       \
         Nums blocks_number = t1.core->blocks.size();                                            \
         for (Nums i = 0; i < blocks_number; i++) {                                              \
            ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();                      \
            const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();                \
            Size block_size = t1.core->blocks[i].size;                                           \
            for (Size j = 0; j < block_size; j++) {                                              \
               EVAL2;                                                                            \
            }                                                                                    \
         }                                                                                       \
      }                                                                                          \
      return t1;                                                                                 \
   }                                                                                             \
   template<                                                                                     \
         class ScalarType1,                                                                      \
         class ScalarType2,                                                                      \
         class Symmetry,                                                                         \
         class = std::enable_if_t<is_scalar_v<ScalarType2>>>                                     \
   Tensor<ScalarType1, Symmetry>& OP(Tensor<ScalarType1, Symmetry>& t1, const ScalarType2& n2) { \
      return OP(t1, Tensor<ScalarType2, Symmetry>{n2});                                          \
   }
   DEF_SCALAR_OP(operator+=, a[j] += y, a[j] += b[j])
   DEF_SCALAR_OP(operator-=, a[j] -= y, a[j] -= b[j])
   DEF_SCALAR_OP(operator*=, a[j] *= y, a[j] *= b[j])
   DEF_SCALAR_OP(operator/=, a[j] /= y, a[j] /= b[j])
#undef DEF_SCALAR_OP

   template<class T>
   void raw_write(std::ostream& out, const T* data, Size number = 1) {
      out.write(reinterpret_cast<const char*>(data), sizeof(T) * number);
   }
   template<class T>
   void raw_read(std::istream& in, T* data, Size number = 1) {
      in.read(reinterpret_cast<char*>(data), sizeof(T) * number);
   }
   bool is_text_stream(const std::ostream& out) {
      return &out == &std::cout || &out == &std::cerr || &out == &std::clog;
   }

   std::ostream& operator<<(std::ostream& out, const Name& name) {
      if (is_text_stream(out)) {
         try {
            return out << id_to_name.at(name.id);
         } catch (const std::out_of_range& e) {
            return out << "UserDefinedName" << name.id;
         }
      } else {
         raw_write(out, &name.id);
      }
      return out;
   }

   std::istream& operator>>(std::istream& in, Name& name) {
      raw_read(in, &name.id);
      return in;
   }

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec) {
      if (is_text_stream(out)) {
         out << "[";
         bool notFirst = false;
         for (const auto& i : vec) {
            if (notFirst) {
               out << ",";
            }
            notFirst = true;
            out << i;
         }
         out << "]";
      } else {
         for (const auto& i : vec) {
            out << i;
         }
      }
      return out;
   }
   template<class T>
   std::istream& operator>>(std::istream& in, vector<T>& vec) {
      for (auto& i : vec) {
         in >> i;
      }
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   std::istream& operator>>(std::istream& in, NoSymmetry&) {
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s) {
      if (is_text_stream(out)) {
         out << s.z2;
      } else {
         raw_write(out, &s.z2);
      }
      return out;
   }
   std::istream& operator>>(std::istream& in, Z2Symmetry& s) {
      raw_read(in, &s.z2);
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const U1Symmetry& s) {
      if (is_text_stream(out)) {
         out << s.u1;
      } else {
         raw_write(out, &s.u1);
      }
      return out;
   }
   std::istream& operator>>(std::istream& in, U1Symmetry& s) {
      raw_read(in, &s.u1);
      return in;
   }
   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge) {
      if (is_text_stream(out)) {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            out << edge.at(NoSymmetry());
         } else {
            out << "{";
            bool notFirst = false;
            for (const auto& [sym, dim] : edge) {
               if (notFirst) {
                  out << ",";
               }
               notFirst = true;
               out << sym << ":" << dim;
            }
            out << "}";
         }
      } else {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            raw_write(out, &edge.at(NoSymmetry()));
         } else {
            Nums nums = edge.size();
            raw_write(out, &nums);
            for (const auto& [sym, dim] : edge) {
               out << sym;
               raw_write(out, &dim);
            }
         }
      }
      return out;
   }
   template<class Symmetry>
   std::istream& operator>>(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         Size dim;
         raw_read(in, &dim);
         edge[NoSymmetry()] = dim;
      } else {
         Nums nums;
         raw_read(in, &nums);
         edge.clear();
         for (Nums i = 0; i < nums; i++) {
            Symmetry sym;
            Size dim;
            in >> sym;
            raw_read(in, &dim);
            edge[sym] = dim;
         }
      }
      return in;
   }

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Block<ScalarType, Symmetry>& block) {
      if (!is_text_stream(out)) {
         TAT_WARNING("Should not come here");
      }
      out << "{";
      if constexpr (!std::is_same_v<Symmetry, NoSymmetry>) {
         out << "symmetry:[";
         bool notFirst = false;
         for (const auto& i : block.symmetries) {
            if (notFirst) {
               out << ",";
            }
            notFirst = true;
            out << i;
         }
         out << "],";
      }
      out << "size:";
      out << block.size;
      out << ",data:[";
      bool notFirst = false;
      for (Size i = 0; i < block.size; i++) {
         if (notFirst) {
            out << ",";
         }
         notFirst = true;
         out << block.raw_data[i];
      }
      out << "]}";
      return out;
   }

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      if (is_text_stream(out)) {
         out << "{names:";
         out << tensor.names;
         out << ",edges:";
         out << tensor.core->edges;
         out << ",blocks:";
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            out << tensor.core->blocks[0];
         } else {
            out << tensor.core->blocks;
         }
         out << "}";
      } else {
         Rank rank = tensor.names.size();
         raw_write(out, &rank);
         out << tensor.names;
         out << tensor.core->edges;
         for (const auto& i : tensor.core->blocks) {
            raw_write(out, i.raw_data.data(), i.size);
         }
      }
      return out;
   }

   template<class ScalarType, class Symmetry>
   std::istream& operator>>(std::istream& in, Tensor<ScalarType, Symmetry>& tensor) {
      Rank rank;
      raw_read(in, &rank);
      tensor.names.resize(rank);
      in >> tensor.names;
      tensor.name_to_index = construct_name_to_index(tensor.names);
      vector<Edge<Symmetry>> edges(rank);
      in >> edges;
      tensor.core =
            std::make_shared<typename Tensor<ScalarType, Symmetry>::TensorCore>(std::move(edges));
      for (auto& i : tensor.core->blocks) {
         raw_read(in, i.raw_data.data(), i.size);
      }
      return in;
   }

   // TODO: lazy framework
   // 需要考虑深搜不可行的问题
   // 支持inplace操作

   // GPU and so on
} // namespace TAT

#endif
