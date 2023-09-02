/**
 * \file tensor.hpp
 *
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#pragma once
#ifndef TAT_TENSOR_HPP
#define TAT_TENSOR_HPP

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <variant>

#include "../utility/allocator.hpp"
#include "../utility/shared_ptr.hpp"
#include "core.hpp"
#include "edge.hpp"
#include "name.hpp"
#include "symmetry.hpp"

namespace TAT {
   struct RemainCut {
      Size value;
      explicit RemainCut(Size v) : value(v) {}
   };
   struct RelativeCut {
      double value;
      explicit RelativeCut(double v) : value(v) {}
   };
   struct NoCut {};
   /**
    * Used to describle how to cut when doing svd to a tensor
    */
   struct Cut {
      Size remain_cut;
      double relative_cut;

      Cut(Size i, double f) : remain_cut(i), relative_cut(f) {}
      Cut(double f, Size i) : remain_cut(i), relative_cut(f) {}
      Cut() : remain_cut(-1), relative_cut(0) {}
      Cut(Size i) : remain_cut(i), relative_cut(0) {}
      Cut(double f) : remain_cut(-1), relative_cut(f) {}

      [[deprecated("NoCut is deprecated, use Cut directly")]] Cut(NoCut) : Cut() {}
      [[deprecated("RelativeCut is deprecated, use Cut directly")]] Cut(RelativeCut c) : Cut(c.value) {}
      [[deprecated("RemainCut is deprecated, use Cut directly")]] Cut(RemainCut c) : Cut(c.value) {}
   };

   template<typename ScalarType = double, typename Symmetry = Symmetry<>, typename Name = DefaultName>
   struct TensorShape;

   /**
    * Tensor type
    *
    * tensor type contains edge name, edge shape, and tensor content.
    * every edge has a Name as its name, for nom-symmetric tensor, an edge is
    * just a number describing its dimension.
    * for symmetric tensor, an edge is a segment like structure, describing
    * each symmetry's dimension.
    * tensor content is represented as several blocks, for non-symmetric tensor,
    * there is only one block
    *
    * \tparam ScalarType scalar type of tensor content
    * \tparam Symmetry tensor's symmetry
    * \tparam Name name type to distinguish different edge
    */
   template<typename ScalarType = double, typename Symmetry = Symmetry<>, typename Name = DefaultName>
   class Tensor {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>);

      template<typename, typename, typename>
      friend class Tensor;
      template<typename ScalarTypeT, typename SymmetryT, typename NameT>
      friend Tensor<ScalarTypeT, SymmetryT, NameT> contract_without_fuse(
            const Tensor<ScalarTypeT, SymmetryT, NameT>& tensor_1,
            const Tensor<ScalarTypeT, SymmetryT, NameT>& tensor_2,
            const std::unordered_set<std::pair<NameT, NameT>>& contract_pairs);
      template<typename ScalarTypeT, typename NameT>
      friend Tensor<ScalarTypeT, NoSymmetry, NameT> contract_with_fuse(
            const Tensor<ScalarTypeT, NoSymmetry, NameT>& tensor_1,
            const Tensor<ScalarTypeT, NoSymmetry, NameT>& tensor_2,
            const std::unordered_set<std::pair<NameT, NameT>>& contract_pairs,
            const std::unordered_set<NameT>& fuse_names);
      template<typename ScalarTypeT, typename SymmetryT, typename NameT>
      friend Tensor<ScalarTypeT, SymmetryT, NameT>
      trace_without_fuse(const Tensor<ScalarTypeT, SymmetryT, NameT>& tensor, const std::unordered_set<std::pair<NameT, NameT>>& trace_pairs);
      template<typename ScalarTypeT, typename SymmetryT, typename NameT>
      friend Tensor<ScalarTypeT, SymmetryT, NameT> trace_with_fuse(
            const Tensor<ScalarTypeT, SymmetryT, NameT>& tensor,
            const std::unordered_set<std::pair<NameT, NameT>>& trace_pairs,
            const std::unordered_map<NameT, std::pair<NameT, NameT>>& fuse_names);
    public:
      using self_t = Tensor<ScalarType, Symmetry, Name>;
      // common used type alias
      using scalar_t = ScalarType;
      using symmetry_t = Symmetry;
      using name_t = Name;
      using edge_t = Edge<symmetry_t>;
      using core_t = Core<scalar_t, symmetry_t>;

      // tensor data
    private:
      /**
       * name of tensor's edge
       * \see Name
       */
      std::vector<name_t> m_names;
      /**
       * tensor data except name, including edge and block
       * \see Core
       * \note bacause edge rename is very common operation, to avoid copy data, put the remaining data into shared pointer
       */
      detail::shared_ptr<core_t> m_core;

    public:
      // names
      [[nodiscard]] const std::vector<name_t>& names() const {
         return m_names;
      }
      [[nodiscard]] const name_t& names(Rank i) const {
         return m_names[i];
      }

      [[nodiscard]] Rank rank() const {
         return m_names.size();
      }
      [[nodiscard]] auto find_by_name(const name_t& name) const {
         return std::find(m_names.begin(), m_names.end(), name);
      }
      [[nodiscard]] Rank rank_by_name(const name_t& name) const {
         auto where = find_by_name(name);
         if (debug_mode) {
            if (where == m_names.end()) {
               detail::error("No such name in name list");
            }
         }
         return std::distance(m_names.begin(), where);
      }

      // core
      [[nodiscard]] const auto& storage() const {
         return m_core->storage();
      }
      [[nodiscard]] auto& storage() {
         return m_core->storage();
      }

      [[nodiscard]] const std::vector<edge_t>& edges() const {
         return m_core->edges();
      }
      [[nodiscard]] const edge_t& edges(Rank r) const {
         return m_core->edges(r);
      }
      [[nodiscard]] const edge_t& edges(const Name& name) const {
         return edges(rank_by_name(name));
      }

      [[nodiscard]] const mdspan<std::optional<mdspan<scalar_t>>>& blocks() const {
         return m_core->blocks();
      }
      [[nodiscard]] mdspan<std::optional<mdspan<scalar_t>>>& blocks() {
         return m_core->blocks();
      }
      [[nodiscard]] const mdspan<std::optional<mdspan<scalar_t>>>& const_blocks() const {
         return m_core->blocks();
      }
      // int/name -> symmetry/position
      // it maybe used from other tensor function
      template<typename T = symmetry_t, typename A>
      [[nodiscard]] const auto& blocks(const std::vector<T, A>& arg) const {
         return m_core->blocks(arg);
      }
      template<typename T = symmetry_t, typename H, typename K, typename A>
      [[nodiscard]] const auto& blocks(const std::unordered_map<name_t, T, H, K, A>& arg_map) const {
         std::vector<T, typename std::allocator_traits<A>::template rebind_alloc<T>> arg_list;
         arg_list.reserve(rank());
         for (const auto& name : names()) {
            arg_list.push_back(arg_map.at(name));
         }
         return blocks(arg_list);
      }
      template<typename T>
      [[nodiscard]] auto& blocks(const T& arg) {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<mdspan<scalar_t>&>(const_cast<const self_t*>(this)->blocks(arg));
      }
      template<typename T>
      [[nodiscard]] const auto& const_blocks(const T& arg) const {
         return blocks(arg);
      }
      // Two of non const version are needed
      // Since c++ need to try to convert literal value to 2 kind of type(vector<symmetry>, map<name, symmetry>).
      [[nodiscard]] auto& blocks(const std::vector<symmetry_t>& arg) {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<mdspan<scalar_t>&>(const_cast<const self_t*>(this)->blocks(arg));
      }
      [[nodiscard]] const auto& const_blocks(const std::vector<symmetry_t>& arg) const {
         return blocks(arg);
      }
      [[nodiscard]] auto& blocks(const std::unordered_map<name_t, symmetry_t>& arg) {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<mdspan<scalar_t>&>(const_cast<const self_t*>(this)->blocks(arg));
      }
      [[nodiscard]] const auto& const_blocks(const std::unordered_map<name_t, symmetry_t>& arg) const {
         return blocks(arg);
      }

      // get element
      // name/int -> (coord/point/index) const/non-const
      template<typename T, typename A>
      [[nodiscard]] const scalar_t& at(const std::vector<T, A>& arg) const {
         return m_core->at(arg);
      }
      template<typename T, typename H, typename K, typename A>
      [[nodiscard]] const scalar_t& at(const std::unordered_map<name_t, T, H, K, A>& arg_map) const {
         std::vector<T, typename std::allocator_traits<A>::template rebind_alloc<T>> arg_list;
         arg_list.reserve(rank());
         for (const auto& name : names()) {
            arg_list.push_back(arg_map.at(name));
         }
         return at(arg_list);
      }
      template<typename T>
      [[nodiscard]] scalar_t& at(const T& arg) {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<scalar_t&>(const_cast<const self_t*>(this)->at(arg));
      }
      template<typename T>
      [[nodiscard]] const scalar_t& const_at(const T& arg) const {
         return at(arg);
      }

      [[nodiscard]] const scalar_t& at() const {
         if (!scalar_like()) {
            detail::error("Try to get the only element of t he tensor which contains more than one element");
         }
         return storage().front();
      }
      [[nodiscard]] scalar_t& at() {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<scalar_t&>(const_cast<const self_t*>(this)->at());
      }
      [[nodiscard]] const scalar_t& const_at() const {
         return at();
      }

      template<typename T>
      using map_from_name = std::unordered_map<name_t, T>;
#define TAT_DEFINE_TENSOR_AT(ARG) \
   [[nodiscard]] const scalar_t& at(const ARG& arg) const { \
      return at<>(arg); \
   } \
   [[nodiscard]] scalar_t& at(const ARG& arg) { \
      return at<>(arg); \
   } \
   [[nodiscard]] const scalar_t& const_at(const ARG& arg) const { \
      return const_at<>(arg); \
   }
      // coord may be ambiguous to point, because position may be ambiguous to symmetry
      // TAT_DEFINE_TENSOR_AT(std::vector<typename edge_t::coord_t>)
      TAT_DEFINE_TENSOR_AT(std::vector<typename edge_t::index_t>)
      TAT_DEFINE_TENSOR_AT(std::vector<typename edge_t::point_t>)
      // TAT_DEFINE_TENSOR_AT(map_from_name<typename edge_t::coord_t>)
      TAT_DEFINE_TENSOR_AT(map_from_name<typename edge_t::index_t>)
      TAT_DEFINE_TENSOR_AT(map_from_name<typename edge_t::point_t>)
#undef TAT_DEFINE_TENSOR_AT

    private:
      /**
       * Check list of names is a valid and the rank is correct
       */
      void check_valid_name() const {
         if (rank() != edges().size()) {
            detail::error("Wrong name list length which no equals to expected length");
         }
         for (auto i = 0; i < rank(); i++) {
            for (auto j = i + 1; j < rank(); j++) {
               if (names(i) == names(j)) {
                  detail::error("Duplicated names in name list");
               }
            }
         }
      }

      // shape
    public:
      /**
       * Get tensor shape to print, used when you don't want to know value of the tensor
       */
      [[nodiscard]] TensorShape<ScalarType, Symmetry, Name> shape() const {
         return {this};
      }

      // constructors
      // There are many method to construct edge, so it is not proper to use initializer list
    public:
      /**
       * Initialize tensor with tensor edge name and tensor edge shape, blocks will be generated by edges
       *
       * \param input_names edge name
       * \param input_edges edge shape
       * \see Core
       */
      Tensor(std::vector<Name> input_names, std::vector<Edge<Symmetry>> input_edges) :
            m_names(std::move(input_names)),
            m_core(detail::shared_ptr<core_t>::make(std::move(input_edges))) {
         if constexpr (debug_mode) {
            check_valid_name();
         }
      }

      Tensor() : Tensor(1) {}
      Tensor(const Tensor& other) = default;
      Tensor(Tensor&& other) = default;
      Tensor& operator=(const Tensor& other) = default;
      Tensor& operator=(Tensor&& other) = default;
      ~Tensor() = default;

      /**
       * Create a high rank tensor but which only contains one element
       *
       * \param number the only element
       * \param input_names edge name
       * \param edge_symmetry the symmetry for every edge, if valid
       * \param edge_arrow the fermi arrow for every edge, if valid
       */
      explicit Tensor(
            scalar_t number,
            std::vector<Name> input_names = {},
            const std::vector<symmetry_t>& edge_symmetry = {},
            const std::vector<Arrow>& edge_arrow = {}) :
            m_names(std::move(input_names)),
            m_core(detail::shared_ptr<core_t>::make(get_edge_from_edge_symmetry_and_arrow(edge_symmetry, edge_arrow, rank()))) {
         if constexpr (debug_mode) {
            check_valid_name();
         }
         at() = number;
      }

    private:
      [[nodiscard]] static std::vector<Edge<Symmetry>>
      get_edge_from_edge_symmetry_and_arrow(const std::vector<Symmetry>& edge_symmetry, const std::vector<Arrow>& edge_arrow, Rank rank) {
         // used in Tensor(ScalarType, ...)
         if constexpr (Symmetry::length == 0) {
            return std::vector<Edge<Symmetry>>(rank, {1});
         } else {
            auto result = std::vector<Edge<Symmetry>>();
            result.reserve(rank);
            if constexpr (Symmetry::is_fermi_symmetry) {
               for (auto i = 0; i < rank; i++) {
                  result.push_back({{{edge_symmetry[i], 1}}, edge_arrow[i]});
               }
            } else {
               for (auto i = 0; i < rank; i++) {
                  result.push_back({{{edge_symmetry[i], 1}}});
               }
            }
            return result;
         }
      }

    public:
      [[nodiscard]] bool scalar_like() const {
         return storage().size() == 1;
      }

      /**
       * Get the only element from a tensor which contains only one element
       */
      explicit operator scalar_t() const {
         if (storage().size() == 0) {
            // sometimes it is useful
            return 0;
         } else {
            return at();
         }
      }

      // elementwise operators
    public:
      /**
       * Do the same operator to the every value element of the tensor, inplacely
       * \param function The operator
       */
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>& transform(Function&& function) & {
         acquare_data_ownership("Set tensor shared in transform, copy happened here");
         std::transform(storage().begin(), storage().end(), storage().begin(), std::forward<Function>(function));
         return *this;
      }
      template<typename Function>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& transform(Function&& function) && {
         return std::move(transform(std::forward<Function>(function)));
      }
      template<typename OtherScalarType, typename Function>
      Tensor<ScalarType, Symmetry, Name>& zip_transform(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function) & {
         acquare_data_ownership("Set tensor shared in zip_transform, copy happened here");
         if constexpr (debug_mode) {
            if (rank() != other.rank()) {
               detail::error("Try to do zip_transform on two different rank tensor");
            }
         }
         auto real_other_pointer = &other;
         auto new_other = Tensor<OtherScalarType, Symmetry, Name>();
         if (names() != other.names()) {
            new_other = other.transpose(names());
            real_other_pointer = &new_other;
         }
         const auto& real_other = *real_other_pointer;
         if constexpr (debug_mode) {
            if (edges() != real_other.edges()) {
               detail::error("Try to do zip_transform on two tensors which edges not compatible");
            }
         }
         std::transform(storage().begin(), storage().end(), real_other.storage().begin(), storage().begin(), std::forward<Function>(function));
         return *this;
      }
      template<typename OtherScalarType, typename Function>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& zip_transform(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function) && {
         return std::move(zip_transform(other, std::forward<Function>(function)));
      }

      /**
       * Generate a tensor with the same shape
       * \tparam NewScalarType basic scalar type of the result tensor
       * \return The value of tensor is not initialized
       */
      template<typename NewScalarType = ScalarType>
      [[nodiscard]] Tensor<NewScalarType, Symmetry, Name> same_shape() const {
         return Tensor<NewScalarType, Symmetry, Name>(names(), edges());
      }

      /**
       * Do the same operator to the every value element of the tensor, outplacely
       * \param function The operator
       * \return The result tensor
       * \see same_shape
       */
      template<typename ForceScalarType = void, typename Function>
      [[nodiscard]] auto map(Function&& function) const {
         using DefaultNewScalarType = std::invoke_result_t<Function, ScalarType>;
         using NewScalarType = std::conditional_t<std::is_same_v<void, ForceScalarType>, DefaultNewScalarType, ForceScalarType>;
         auto result = same_shape<NewScalarType>();
         std::transform(storage().begin(), storage().end(), result.storage().begin(), std::forward<Function>(function));
         return result;
      }

      template<typename ForceScalarType = void, typename OtherScalarType, typename Function>
      [[nodiscard]] auto zip_map(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function) const {
         using DefaultNewScalarType = std::invoke_result_t<Function, ScalarType, OtherScalarType>;
         using NewScalarType = std::conditional_t<std::is_same_v<void, ForceScalarType>, DefaultNewScalarType, ForceScalarType>;
         if constexpr (debug_mode) {
            if (rank() != other.rank()) {
               detail::error("Try to do zip_map on two different rank tensor");
            }
         }
         auto real_other_pointer = &other;
         auto new_other = Tensor<OtherScalarType, Symmetry, Name>();
         if (names() != other.names()) {
            new_other = other.transpose(names());
            real_other_pointer = &new_other;
         }
         const auto& real_other = *real_other_pointer;
         if constexpr (debug_mode) {
            if (edges() != real_other.edges()) {
               detail::error("Try to do zip_map on two tensors which edges not compatible");
            }
         }
         auto result = same_shape<NewScalarType>();
         std::transform(storage().begin(), storage().end(), real_other.storage().begin(), result.storage().begin(), std::forward<Function>(function));
         return result;
      }

      /**
       * Tensor deep copy, default copy will share the common data, i.e. the same core
       * \see map
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> copy() const {
         return map([](const ScalarType& x) -> ScalarType {
            return x;
         });
      }

      /**
       * Set value of tensor by a generator elementwisely
       * \param generator Generator accept non argument, and return scalartype
       */
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>& set(Generator&& generator) & {
         acquare_data_ownership("Set tensor shared, copy happened here");
         std::generate(storage().begin(), storage().end(), std::forward<Generator>(generator));
         return *this;
      }
      template<typename Generator>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& set(Generator&& generator) && {
         return std::move(set(std::forward<Generator>(generator)));
      }

      /**
       * Set all the value of the tensor to zero
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& zero() & {
         return set([]() -> ScalarType {
            return 0;
         });
      }
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& zero() && {
         return std::move(zero());
      }

      /**
       * Set the value of tensor as natural number, used for test
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& range(ScalarType first = 0, ScalarType step = 1) & {
         return set([&first, step]() -> ScalarType {
            auto result = first;
            first += step;
            return result;
         });
      }
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& range(ScalarType first = 0, ScalarType step = 1) && {
         return std::move(range(first, step));
      }

      /**
       * Acquare tensor data's ownership, it will copy the core if the core is shared
       * \param message warning message if core is copied
       */
      void acquare_data_ownership(const char* message = "") {
         if (m_core.use_count() != 1) {
            m_core = detail::shared_ptr<Core<ScalarType, Symmetry>>::make(*m_core);
            if (*message != 0) {
               detail::what_if_copy_shared(message);
            }
         }
      }

      /**
       * Change the basic scalar type of the tensor
       */
      template<typename OtherScalarType, typename = std::enable_if_t<is_scalar<OtherScalarType>>>
      [[nodiscard]] Tensor<OtherScalarType, Symmetry, Name> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            return *this; // shallow copy
         } else {
            return map([](ScalarType input) -> OtherScalarType {
               if constexpr (is_complex<ScalarType> && is_real<OtherScalarType>) {
                  return OtherScalarType(input.real());
               } else {
                  return OtherScalarType(input);
               }
            });
         }
      }

      /**
       * Get the norm of the tensor
       * \note Treat the tensor as vector, not the matrix norm or other things
       * \tparam p Get the p-norm of the tensor, if p=-1, that is max absolute value norm, namely inf-norm
       */
      template<int p = 2>
      [[nodiscard]] real_scalar<ScalarType> norm() const {
         real_scalar<ScalarType> result = 0;
         if constexpr (p == -1) {
            // max abs
            for (const auto& number : storage()) {
               if (auto absolute_value = std::abs(number); absolute_value > result) {
                  result = absolute_value;
               }
            }
         } else if constexpr (p == 0) {
            result += real_scalar<ScalarType>(storage().size());
         } else {
            for (const auto& number : storage()) {
               if constexpr (p == 1) {
                  result += std::abs(number);
               } else if constexpr (p == 2) {
                  result += std::norm(number);
               } else {
                  if constexpr (p % 2 == 0 && is_real<ScalarType>) {
                     result += std::pow(number, p);
                  } else {
                     result += std::pow(std::abs(number), p);
                  }
               }
            }
            result = std::pow(result, 1. / p);
         }
         return result;
      }

    public:
      /**
       * Convert symmetry tensor to non-symmetry tensor
       *
       * \note it is dangerous for fermi tensor
       */
      [[nodiscard]] Tensor<ScalarType, NoSymmetry, Name> clear_bose_symmetry() const;

      /**
       * Convert fermionic symmetry tensor to parity symmetry tensor
       *
       * \note it is invalid for bose tensor
       */
      [[nodiscard]] Tensor<ScalarType, ParitySymmetry, Name> clear_fermi_symmetry() const;

      [[nodiscard]] auto clear_symmetry() const {
         if constexpr (symmetry_t::is_fermi_symmetry) {
            return clear_fermi_symmetry();
         } else {
            return clear_bose_symmetry();
         }
      }


    public:
      // Operators
      /**
       * The core method for various edge operations, include split, reverse, merge and transpose
       * \param split_map map describing how to split
       * \param reversed_names set describing how to reverse, only for fermi tensor
       * \param merge_map map describing how to merge
       * \param new_names the result tensor edge order
       * \param apply_parity some operations generate half sign, it controls default behavior whether to apply the sign to this tensor
       * \return the result of all the operations
       *
       * If some few edge is not share the same behavior to default sign apply property, please use the last four argument
       *
       * \note If reversed name not satisfy the merge condition, it will reverse automatically
       * \note For fermi tensor, reverse/split/merge will generate half sign, you need to apply the sign to one of the two tensor
       * \note Since transpose generate a full sign, it will not be controled by apply_parity, it is always valid
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> edge_operator(
            const std::unordered_map<Name, std::vector<std::pair<Name, edge_segments_t<Symmetry>>>>& split_map,
            const std::unordered_set<Name>& reversed_names,
            const std::unordered_map<Name, std::vector<Name>>& merge_map,
            std::vector<Name> new_names, // move into result tensor
            const bool apply_parity = false,
            const std::unordered_set<Name>& parity_exclude_names_split = {},
            const std::unordered_set<Name>& parity_exclude_names_reversed_before_transpose = {},
            const std::unordered_set<Name>& parity_exclude_names_reversed_after_transpose = {},
            const std::unordered_set<Name>& parity_exclude_names_merge = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               split_map,
               reversed_names,
               merge_map,
               std::move(new_names),
               apply_parity,
               parity_exclude_names_split,
               parity_exclude_names_reversed_before_transpose,
               parity_exclude_names_reversed_after_transpose,
               parity_exclude_names_merge,
               {});
         // last argument only used in svd, Name -> Symmetry -> Size
         // it is not proper to expose to users
      }

    private:
      template<
            typename A = empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segments_t<Symmetry>>>>>,
            typename B = empty_list<Name>,
            typename C = empty_list<std::pair<Name, empty_list<Name>>>,
            typename D = empty_list<Name>,
            typename E = empty_list<Name>,
            typename F = empty_list<Name>,
            typename G = empty_list<Name>,
            typename H = empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>>
      [[nodiscard]] Tensor<scalar_t, symmetry_t, name_t> edge_operator_implement(
            const A& split_map,
            const B& reversed_names,
            const C& merge_map,
            std::vector<Name> new_names,
            const bool apply_parity,
            const D& parity_exclude_names_split,
            const E& parity_exclude_names_reversed_before_transpose,
            const F& parity_exclude_names_reversed_after_transpose,
            const G& parity_exclude_names_merge,
            const H& edges_and_symmetries_to_cut_before_all) const;

    public:
      /**
       * Rename the edge name of tensor
       * \param dictionary the map of the plan for renaming edge name
       * \return A tensor after renaming, share the core with the original tensor
       */
      template<typename ResultName = Name, typename = std::enable_if_t<is_name<ResultName>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, ResultName> edge_rename(const std::unordered_map<Name, ResultName>& dictionary) const {
         if constexpr (debug_mode) {
            for (const auto& [name, new_name] : dictionary) {
               if (auto found = find_by_name(name); found == names().end()) {
                  detail::error("Name missing in edge_rename");
               }
            }
         }
         auto result = Tensor<ScalarType, Symmetry, ResultName>();
         result.m_core = m_core; // shallow copy
         result.m_names.reserve(rank());
         std::transform(names().begin(), names().end(), std::back_inserter(result.m_names), [&dictionary](const Name& name) {
            if (auto position = dictionary.find(name); position == dictionary.end()) {
               if constexpr (std::is_same_v<ResultName, Name>) {
                  return name;
               } else {
                  detail::error("New names not found in edge_rename which change type of name");
               }
            } else {
               return position->second;
            }
         });
         return result;
      }

      /**
       * Transpose the tensor
       * \param target_names edge name order after transpose
       * \return tensor transposed
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> transpose(std::vector<Name> target_names) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         if (names() == target_names) {
            return *this; // shallow copy
         }
         return edge_operator_implement({}, {}, {}, std::move(target_names), false, {}, {}, {}, {}, {});
      }

      /**
       * Reverse fermi arrow of some edge for fermi tensor
       * \param reversed_names reversed name set
       * \param apply_parity whether to apply sign by default
       * \param parity_exclude_names set of edge which apply sign differently with default behavior
       * \return tensor with edge reversed
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> reverse_edge(
            const std::unordered_set<Name>& reversed_names,
            bool apply_parity = false,
            const std::unordered_set<Name>& parity_exclude_names = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement({}, reversed_names, {}, names(), apply_parity, {}, parity_exclude_names, {}, {}, {});
      }

      /**
       * Merge some edge of a tensor
       * \param merge a map describing how to merge
       * \note the strategy to determine the result edge is to move each group of merged edge to the last edge of this merge group
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> merge_edge(
            const std::unordered_map<Name, std::vector<Name>>& merge,
            bool apply_parity = false,
            const std::unordered_set<Name>& parity_exclude_names_merge = {},
            const std::unordered_set<Name>& parity_exclude_names_reverse = {}) const;

      /**
       * Split some edge of a tensor
       * \param split a map describing how to split
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> split_edge(
            const std::unordered_map<Name, std::vector<std::pair<Name, edge_segments_t<Symmetry>>>>& split,
            bool apply_parity = false,
            const std::unordered_set<Name>& parity_exclude_names_split = {}) const;

      // Contract
    private:
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name> contract_implement(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
            const std::unordered_set<Name>& fuse_names);
    public:
      /**
       * Calculate product of two tensor
       * \param tensor_1 tensor 1
       * \param tensor_2 tensor 2
       * \param contract_pairs set of edge name pair to contract
       * \param fuse_names set of edge name to fuse
       * \return the result tensor
       */
      template<typename ScalarType1, typename ScalarType2, typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2>>>
      [[nodiscard]] static auto contract(
            const Tensor<ScalarType1, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType2, Symmetry, Name>& tensor_2,
            const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
            const std::unordered_set<Name>& fuse_names = {}) {
         using ResultScalarType = std::common_type_t<ScalarType1, ScalarType2>;
         using ResultTensor = Tensor<ResultScalarType, Symmetry, Name>;
         // Maybe shallow copy happened here
         return ResultTensor::contract_implement(
               tensor_1.template to<ResultScalarType>(),
               tensor_2.template to<ResultScalarType>(),
               contract_pairs,
               fuse_names);
      }

      template<typename OtherScalarType, typename = std::enable_if_t<is_scalar<OtherScalarType>>>
      [[nodiscard]] auto contract(
            const Tensor<OtherScalarType, Symmetry, Name>& tensor_2,
            const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
            const std::unordered_set<Name>& fuse_names = {}) const {
         return contract(*this, tensor_2, contract_pairs, fuse_names);
      }

      /**
       * Get the conjugated tensor
       * \note for symmetry tensor, every symmetry is transformed to -symmetry,
       * for fermion tensor, arrow is reversed, for complex tensor value got conjugated
       * \param trivial_metric apply a trivial metric when conjugate the tensor.
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> conjugate(bool trivial_metric = false) const;

      /**
       * Set the tensor as identity inplacely
       * \param pairs pair set describing how to treat the tensor as matrix
       */
      Tensor<ScalarType, Symmetry, Name>& identity(const std::unordered_set<std::pair<Name, Name>>& pairs) &;

      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& identity(const std::unordered_set<std::pair<Name, Name>>& pairs) && {
         return std::move(identity(pairs));
      }

      /**
       * Get the tensor exponential
       * \param pairs pair set describing how to treat the tensor as matrix
       * \param step iteration step
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> exponential(const std::unordered_set<std::pair<Name, Name>>& pairs, int step = 2) const;

      /**
       * Get trace of tensor
       * \param trace_pairs pair set describing how to trace the tensor
       * \param fuse_names the edges need to be fused
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      trace(const std::unordered_set<std::pair<Name, Name>>& trace_pairs,
            const std::unordered_map<Name, std::pair<Name, Name>>& fuse_names = {}) const;

      /**
       * SVD result type
       */
      struct svd_result {
         Tensor<ScalarType, Symmetry, Name> U;
         Tensor<ScalarType, Symmetry, Name> S;
         Tensor<ScalarType, Symmetry, Name> V;
      };

      /**
       * QR result type
       */
      struct qr_result {
         Tensor<ScalarType, Symmetry, Name> Q;
         Tensor<ScalarType, Symmetry, Name> R;
      };

      /**
       * Calculate SVD of the tensor
       * \param free_names_u U tensor free name after SVD
       * \param common_name_u U tensor new name after SVD
       * \param common_name_v V tensor new name after SVD
       * \param singular_name_u S tensor edge name connected to tensor U
       * \param singular_name_v S tensor edge name connected to tensor V
       * \param cut How to cut bond dimension during SVD
       * \return SVD result
       * \see svd_result
       */
      [[nodiscard]] svd_result
      svd(const std::unordered_set<Name>& free_names_u,
          const Name& common_name_u,
          const Name& common_name_v,
          const Name& singular_name_u,
          const Name& singular_name_v,
          Cut cut = Cut()) const;

      /**
       * Calculate QR of the tensor
       * \param free_names_direction specify what tensor the free_name_set means, it can be 'Q' or 'R'
       * \param free_name one of tensor Q or tensor R free name after QR
       * \param common_name_q Q tensor new name after QR
       * \param common_name_r R tensor new name after QR
       * \return QR result
       * \see qr_result
       */
      [[nodiscard]] qr_result
      qr(char free_names_direction, const std::unordered_set<Name>& free_names, const Name& common_name_q, const Name& common_name_r) const;

      /**
       * Expand a one dimensional edge of a tensor to several wider edge
       * \param configure The way to expand edge. It is a map from the name of the new edge to a pair containing new index and the new full edge.
       * \param old_name the original edge to be expanded, it should be a one dimensional edge.
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> expand(
            const std::unordered_map<Name, std::pair<Size, Edge<Symmetry>>>& configure,
            const Name& old_name = InternalName<Name>::No_Old_Name) const;
      /**
       * Shrink several edge of a tensor to a one dimensional edge.
       * \param configure The way to shrink edges. It is a map from the name of the old edge to the kept index.
       * \param new_name The name of the new edge created by shrinking.
       * \param arrow The fermionic arrow of the new edge for fermionic tensor.
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      shrink(const std::unordered_map<Name, Size>& configure, const Name& new_name = InternalName<Name>::No_New_Name, Arrow arrow = false) const;

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Tensor<ScalarType, Symmetry, Name>& load(const std::string&) &;
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };

      void _block_order_v0_to_v1() {
         m_core->_block_order_v0_to_v1();
      }
   };

   namespace detail {
      template<typename T>
      struct is_tensor_helper : std::false_type {};
      template<typename A, typename B, typename C>
      struct is_tensor_helper<Tensor<A, B, C>> : std::true_type {};
   } // namespace detail
   template<typename T>
   constexpr bool is_tensor = detail::is_tensor_helper<T>::value;

   template<typename ScalarType1, typename ScalarType2, typename Symmetry, typename Name>
   [[nodiscard]] auto contract(
         const Tensor<ScalarType1, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType2, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_pairs) {
      return tensor_1.contract(tensor_2, contract_pairs);
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   struct TensorShape {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>);

      const Tensor<ScalarType, Symmetry, Name>* owner;
   };

   // TODO quasi tensor (middle value between edge_operator)
} // namespace TAT
#endif
