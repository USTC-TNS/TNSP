/**
 * \file tensor.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
   namespace detail {
      // set and map have method `find`
      template<typename Container>
      using find_checker = decltype(&Container::find);
      template<typename Container>
      constexpr bool have_find_v = is_detected_v<find_checker, Container>;

      template<typename Set, typename Key>
      auto fake_set_find(Set& v, const Key& k) {
         if constexpr (have_find_v<Set>) {
            return v.find(k);
         } else {
            auto result = std::lower_bound(v.begin(), v.end(), k);
            if (result == v.end()) {
               return v.end();
            } else if (k == *result) {
               return result;
            } else {
               return v.end();
            }
         }
      }

      /**
       * If given a Key, return itself, else return a.first;
       *
       * It is used in fake_map find.
       */
      template<typename Key, typename A>
      const auto& get_key(const A& a) {
         if constexpr (std::is_same_v<remove_cvref_t<Key>, remove_cvref_t<A>>) {
            return a;
         } else {
            return std::get<0>(a);
         }
      }

      template<bool is_range, typename Map, typename Key>
      auto fake_map_find(Map& v, const Key& k) {
         if constexpr (have_find_v<Map>) {
            // TODO what if have_find and is_range
            return v.find(k);
         } else {
            auto result = std::lower_bound(v.begin(), v.end(), k, [](const auto& a, const auto& b) {
               if constexpr (is_range) {
                  return std::lexicographical_compare(get_key<Key>(a).begin(), get_key<Key>(a).end(), get_key<Key>(b).begin(), get_key<Key>(b).end());
               } else {
                  return get_key<Key>(a) < get_key<Key>(b);
               }
            });
            if (result == v.end()) {
               // result may be un dereferencable
               return v.end();
            } else {
               if constexpr (is_range) {
                  if (std::equal(std::get<0>(*result).begin(), std::get<0>(*result).end(), k.begin(), k.end())) {
                     return result;
                  }
               } else {
                  if (std::get<0>(*result) == k) {
                     return result;
                  }
               }
               return v.end();
            }
         }
      }
   } // namespace detail

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
    *
    * Is one of RemainCut, RelativeCut and NoCut
    */
   using Cut = std::variant<RemainCut, RelativeCut, NoCut>;

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
   struct Tensor {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>);

      using self_t = Tensor<ScalarType, Symmetry, Name>;
      // common used type alias
      using scalar_t = ScalarType;
      using symmetry_t = Symmetry;
      using name_t = Name;
      using edge_t = Edge<Symmetry>;
      using core_t = Core<ScalarType, Symmetry>;

      // tensor data
      // names
      /**
       * name of tensor's edge
       * \see Name
       */
      std::vector<Name> names;

      Rank get_rank() const {
         return names.size();
      }

      auto find_rank_from_name(const Name& name) const {
         return std::find(names.begin(), names.end(), name);
      }

      Rank get_rank_from_name(const Name& name) const {
         auto where = find_rank_from_name(name);
         if (where == names.end()) {
            detail::error("No such name in name list");
         }
         return std::distance(names.begin(), where);
      }

      /**
       * Check list of names is a valid and the rank is correct
       */
      bool check_valid_name() {
         auto rank = core->edges.size();
         if (names.size() != rank) {
            detail::error("Wrong name list length which no equals to expected length");
            return false;
         }
         for (auto i = names.begin(); i != names.end(); ++i) {
            for (auto j = std::next(i); j != names.end(); ++j) {
               if (*i == *j) {
                  detail::error("Duplicated names in name list");
                  return false;
               }
            }
         }
         return true;
      }

      // core
      /**
       * tensor data except name, including edge and block
       * \see Core
       * \note bacause edge rename is very common operation, to avoid copy data, put the remaining data into shared pointer
       */
      detail::shared_ptr<core_t> core;

      // shape
      /**
       * Get tensor shape to print, used when you don't want to know value of the tensor
       */
      TensorShape<ScalarType, Symmetry, Name> shape() const {
         return {this};
      }

      // constructors
      // There are many method to construct edge, so it is not proper to use initializer list
      /**
       * Initialize tensor with tensor edge name and tensor edge shape, blocks will be generated by edges
       *
       * \param names_init edge name
       * \param edges_init edge shape
       * \see Core
       */
      Tensor(std::vector<Name> names_init, std::vector<Edge<Symmetry>> edges_init) :
            names(std::move(names_init)),
            core(detail::shared_ptr<core_t>::make(std::move(edges_init))) {
         if constexpr (debug_mode) {
            check_valid_name();
         }
      }

      Tensor() : Tensor(1){};
      Tensor(const Tensor& other) = default;
      Tensor(Tensor&& other) noexcept = default;
      Tensor& operator=(const Tensor& other) = default;
      Tensor& operator=(Tensor&& other) noexcept = default;
      ~Tensor() = default;

      /**
       * Create a high rank tensor but which only contains one element
       *
       * \param number the only element
       * \param names_init edge name
       * \param edge_symmetry the symmetry for every edge, if valid
       * \param edge_arrow the fermi arrow for every edge, if valid
       */
      explicit Tensor(
            ScalarType number,
            std::vector<Name> names_init = {},
            const std::vector<Symmetry>& edge_symmetry = {},
            const std::vector<Arrow>& edge_arrow = {}) :
            names(std::move(names_init)),
            core(detail::shared_ptr<core_t>::make(get_edge_from_edge_symmetry_and_arrow(edge_symmetry, edge_arrow, names.size()))) {
         if constexpr (debug_mode) {
            check_valid_name();
         }
         at() = number;
      }

      [[nodiscard]] bool scalar_like() const {
         return storage().size() == 1;
      }

      /**
       * Get the only element from a tensor which contains only one element
       */
      explicit operator ScalarType() const {
         if (storage().size() == 0) {
            // sometimes it is useful
            return 0;
         } else {
            return const_at();
         }
      }

      [[nodiscard]] static auto
      get_edge_from_edge_symmetry_and_arrow(const std::vector<Symmetry>& edge_symmetry, const std::vector<Arrow>& edge_arrow, Rank rank) {
         // used in Tensor(ScalarType, ...)
         if constexpr (Symmetry::length == 0) {
            return std::vector<Edge<Symmetry>>(rank, {1});
         } else {
            auto result = std::vector<Edge<Symmetry>>();
            result.reserve(rank);
            if constexpr (Symmetry::is_fermi_symmetry) {
               for (auto [symmetry, arrow] = std::tuple{edge_symmetry.begin(), edge_arrow.begin()}; symmetry < edge_symmetry.end();
                    ++symmetry, ++arrow) {
                  result.push_back({{{*symmetry, 1}}, *arrow});
               }
            } else {
               for (auto symmetry = edge_symmetry.begin(); symmetry < edge_symmetry.end(); ++symmetry) {
                  result.push_back({{{*symmetry, 1}}});
               }
            }
            return result;
         }
      }

      // elementwise operators
      /**
       * Do the same operator to the every value element of the tensor, inplacely
       * \param function The operator
       */
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>& transform(Function&& function) & {
         acquare_data_ownership("Set tensor shared, copy happened here");
         std::transform(storage().begin(), storage().end(), storage().begin(), function);
         return *this;
      }
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>&& transform(Function&& function) && {
         return std::move(transform(std::forward<Function>(function)));
      }
      template<typename OtherScalarType, typename Function, typename Missing>
      Tensor<ScalarType, Symmetry, Name>&
      zip_transform(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function, Missing&& missing) & {
         acquare_data_ownership("Set tensor shared in zip_transform, copy happened here");
         if (get_rank() != other.get_rank()) {
            detail::error("Try to do zip_transform on two different rank tensor");
         }
         auto real_other_pointer = &other;
         auto new_other = Tensor<OtherScalarType, Symmetry, Name>();
         if (names != other.names) {
            new_other = other.transpose(names);
            real_other_pointer = &new_other;
         }
         const auto& real_other = *real_other_pointer;
         if (core->edges != real_other.core->edges) {
            Nums common_block_number = 0;
            for (auto& [symmetries, block] : core->blocks) {
               if (const auto found = detail::fake_map_find<true>(real_other.core->blocks, symmetries); found != real_other.core->blocks.end()) {
                  // check shape
                  if constexpr (debug_mode) {
                     for (auto i = 0; i < get_rank(); i++) {
                        if (edges(i).get_dimension_from_symmetry(symmetries[i]) != real_other.edges(i).get_dimension_from_symmetry(symmetries[i])) {
                           detail::error("Try to do zip_transform on two tensors which edges not compatible");
                        }
                     }
                  }
                  // call function
                  std::transform(block.begin(), block.end(), found->second.begin(), block.begin(), function);
                  common_block_number++;
               } else {
                  // call missing
                  std::transform(block.begin(), block.end(), block.begin(), missing);
               }
            }
            if (common_block_number != real_other.core->blocks.size()) {
               detail::error("Try to do zip_transform on a tensor which has missing block");
            }
         } else {
            std::transform(storage().begin(), storage().end(), real_other.storage().begin(), storage().begin(), function);
         }
         return *this;
      }
      template<typename OtherScalarType, typename Function, typename Missing>
      Tensor<ScalarType, Symmetry, Name>&&
      zip_transform(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function, Missing&& missing) && {
         return std::move(zip_transform(other, std::forward<Function>(function), std::forward<Missing>(missing)));
      }

      /**
       * Generate a tensor with the same shape
       * \tparam NewScalarType basic scalar type of the result tensor
       * \return The value of tensor is not initialized
       */
      template<typename NewScalarType = ScalarType>
      [[nodiscard]] Tensor<NewScalarType, Symmetry, Name> same_shape() const {
         return Tensor<NewScalarType, Symmetry, Name>(names, core->edges);
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
         std::transform(storage().begin(), storage().end(), result.storage().begin(), function);
         return result;
      }

      template<typename ForceScalarType = void, typename OtherScalarType, typename Function>
      [[nodiscard]] auto zip_map(const Tensor<OtherScalarType, Symmetry, Name>& other, Function&& function) const {
         using DefaultNewScalarType = std::invoke_result_t<Function, ScalarType, OtherScalarType>;
         using NewScalarType = std::conditional_t<std::is_same_v<void, ForceScalarType>, DefaultNewScalarType, ForceScalarType>;
         if (get_rank() != other.get_rank()) {
            detail::error("Try to do zip_map on two different rank tensor");
         }
         auto real_other_pointer = &other;
         auto new_other = Tensor<OtherScalarType, Symmetry, Name>();
         if (names != other.names) {
            new_other = other.transpose(names);
            real_other_pointer = &new_other;
         }
         const auto& real_other = *real_other_pointer;
         if (core->edges != real_other.core->edges) {
            detail::error("Try to do zip_map on two tensors which edges not compatible");
         }
         auto result = same_shape<NewScalarType>();
         std::transform(storage().begin(), storage().end(), real_other.storage().begin(), result.storage().begin(), function);
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
         std::generate(storage().begin(), storage().end(), generator);
         return *this;
      }
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>&& set(Generator&& generator) && {
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
      Tensor<ScalarType, Symmetry, Name>&& zero() && {
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
      Tensor<ScalarType, Symmetry, Name>&& range(ScalarType first = 0, ScalarType step = 1) && {
         return std::move(range(first, step));
      }

      /**
       * Acquare tensor data's ownership, it will copy the core if the core is shared
       * \param message warning message if core is copied
       */
      void acquare_data_ownership(const char* message = "") {
         if (core.use_count() != 1) {
            core = detail::shared_ptr<Core<ScalarType, Symmetry>>::make(*core);
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
            return *this;
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

      // get element
      // name/int -> (point/index) const&/&
#define TAT_DEFINE_TENSOR_AT(...) \
   [[nodiscard]] const ScalarType& at(const __VA_ARGS__& position) const& { \
      return const_at(position); \
   } \
   [[nodiscard]] ScalarType& at(const __VA_ARGS__& position)& { \
      acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference"); \
      return const_cast<ScalarType&>(const_cast<const self_t*>(this)->const_at(position)); \
   } \
   [[nodiscard]] const ScalarType& const_at(const __VA_ARGS__& position) const& { \
      return get_item(position); \
   }
      TAT_DEFINE_TENSOR_AT(std::unordered_map<Name, std::pair<Symmetry, Size>>)
      TAT_DEFINE_TENSOR_AT(std::unordered_map<Name, Size>)
      TAT_DEFINE_TENSOR_AT(std::vector<std::pair<Symmetry, Size>>)
      TAT_DEFINE_TENSOR_AT(std::vector<Size>)
#undef TAT_DEFINE_TENSOR_AT
      [[nodiscard]] const ScalarType& at() const& {
         return const_at();
      }
      [[nodiscard]] ScalarType& at() & {
         acquare_data_ownership("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         return const_cast<ScalarType&>(const_cast<const self_t*>(this)->const_at());
      }
      [[nodiscard]] const ScalarType& const_at() const& {
         if (!scalar_like()) {
            detail::error("Try to get the only element of t he tensor which contains more than one element");
         }
         return storage().front();
      }

      template<typename PositionType>
      [[nodiscard]] const ScalarType& get_item(const PositionType& position) const&;

      /**
       * Convert symmetry tensor to non-symmetry tensor
       *
       * \note it is dangerous for fermi tensor
       */
      Tensor<ScalarType, NoSymmetry, Name> clear_symmetry() const;

      const auto& storage() const& {
         return core->storage;
      }
      auto& storage() & {
         return core->storage;
      }

      const Edge<Symmetry>& edges(Rank r) const& {
         return core->edges[r];
      }
      const Edge<Symmetry>& edges(const Name& name) const& {
         return edges(get_rank_from_name(name));
      }
      Edge<Symmetry>& edges(Rank r) & {
         return const_cast<Edge<Symmetry>&>(const_cast<const self_t*>(this)->edges(r));
      }
      Edge<Symmetry>& edges(const Name& name) & {
         return const_cast<Edge<Symmetry>&>(const_cast<const self_t*>(this)->edges(name));
      }

      // int/name -> symmetry
      template<typename SymmetryList = std::vector<Symmetry>>
      const typename core_t::content_vector_t& blocks(const SymmetryList& symmetry_list) const& {
         // it maybe used from other tensor function
         auto found = detail::fake_map_find<true>(core->blocks, symmetry_list);
         if (found == core->blocks.end()) {
            detail::error("No such symmetry block in the tensor");
         }
         return found->second;
      }
      const typename core_t::content_vector_t& blocks(const std::unordered_map<Name, Symmetry>& symmetry_map) const& {
         std::vector<Symmetry> symmetry_list;
         symmetry_list.reserve(get_rank());
         for (const auto& name : names) {
            symmetry_list.push_back(symmetry_map.at(name));
         }
         return blocks(symmetry_list);
      }
      template<typename SymmetryList = std::vector<Symmetry>>
      typename core_t::content_vector_t& blocks(const SymmetryList& symmetry_list) & {
         return const_cast<typename core_t::content_vector_t&>(const_cast<const self_t*>(this)->blocks(symmetry_list));
      }
      typename core_t::content_vector_t& blocks(const std::unordered_map<Name, Symmetry>& symmetry_map) & {
         return const_cast<typename core_t::content_vector_t&>(const_cast<const self_t*>(this)->blocks(symmetry_map));
      }

      // Operators
      /**
       * The core method for various edge operations, include split, reverse, merge and transpose
       * \param split_map map describing how to split
       * \param reversed_name set describing how to reverse, only for fermi tensor
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
            const std::unordered_map<Name, std::vector<std::pair<Name, edge_segment_t<Symmetry>>>>&
                  split_map, // order of edge symmetry is specify here
            const std::set<Name>& reversed_name,
            const std::unordered_map<Name, std::vector<Name>>& merge_map, // if you want, you can reorder the edge symemtry easily after edge operator
            std::vector<Name> new_names,                                  // move into result tensor
            const bool apply_parity = false,
            const std::set<Name>& parity_exclude_name_split = {},
            const std::set<Name>& parity_exclude_name_reversed_before_transpose = {},
            const std::set<Name>& parity_exclude_name_reversed_after_transpose = {},
            const std::set<Name>& parity_exclude_name_merge = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               split_map,
               reversed_name,
               merge_map,
               std::move(new_names),
               apply_parity,
               parity_exclude_name_split,
               parity_exclude_name_reversed_before_transpose,
               parity_exclude_name_reversed_after_transpose,
               parity_exclude_name_merge,
               empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
         // last argument only used in svd, Name -> Symmetry -> Size
         // it is not proper to expose to users
      }

      template<typename A, typename B, typename C, typename D, typename E, typename F, typename G, typename H>
      [[nodiscard]] auto edge_operator_implement(
            const A& split_map,
            const B& reversed_name,
            const C& merge_map,
            std::vector<Name> new_names,
            const bool apply_parity,
            const D& parity_exclude_name_split,
            const E& parity_exclude_name_reversed_before_transpose,
            const F& parity_exclude_name_reversed_after_transpose,
            const G& parity_exclude_name_merge,
            const H& edge_and_symmetries_to_cut_before_all) const;

      /**
       * Rename the edge name of tensor
       * \param dictionary the map of the plan for renaming edge name
       * \return A tensor after renaming, share the core with the original tensor
       */
      template<typename ResultName = Name, typename = std::enable_if_t<is_name<ResultName>>>
      [[nodiscard]] auto edge_rename(const std::unordered_map<Name, ResultName>& dictionary) const;

      /**
       * Transpose the tensor
       * \param target_names edge name order after transpose
       * \return tensor transposed
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> transpose(std::vector<Name> target_names) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         if (names == target_names) {
            return *this;
         }
         return edge_operator_implement(
               empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, empty_list<Name>>>(),
               std::move(target_names),
               false,
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
      }

      /**
       * Reverse fermi arrow of some edge for fermi tensor
       * \param reversed_name reversed name set
       * \param apply_parity whether to apply sign by default
       * \param parity_exclude_name set of edge which apply sign differently with default behavior
       * \return tensor with edge reversed
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      reverse_edge(const std::set<Name>& reversed_name, bool apply_parity = false, const std::set<Name>& parity_exclude_name = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
               reversed_name,
               empty_list<std::pair<Name, empty_list<Name>>>(),
               names,
               apply_parity,
               empty_list<Name>(),
               parity_exclude_name,
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
      }

      /**
       * Merge some edge of a tensor
       * \param merge a map describing how to merge
       * \note the strategy to determine the result edge is to move each group of merged edge to the last edge of this merge group
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> merge_edge(
            const std::unordered_map<Name, std::vector<Name>>& merge,
            bool apply_parity = false,
            const std::set<Name>&& parity_exclude_name_merge = {},
            const std::set<Name>& parity_exclude_name_reverse = {}) const;

      /**
       * Split some edge of a tensor
       * \param split a map describing how to split
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> split_edge(
            const std::unordered_map<Name, std::vector<std::pair<Name, edge_segment_t<Symmetry>>>>& split,
            bool apply_parity = false,
            const std::set<Name>& parity_exclude_name_split = {}) const;

      // Contract
      // maybe calculate tensor product directly without transpose, but it is very hard
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name> contract_implement(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            const std::set<std::pair<Name, Name>>& contract_names,
            const std::set<Name>& fuse_names);

      /**
       * Calculate product of two tensor
       * \param tensor_1 tensor 1
       * \param tensor_2 tensor 2
       * \param contract_names set of edge name pair to contract
       * \param fuse_names set of edge name to fuse
       * \return the result tensor
       */
      template<typename ScalarType1, typename ScalarType2, typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2>>>
      [[nodiscard]] static auto contract(
            const Tensor<ScalarType1, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType2, Symmetry, Name>& tensor_2,
            const std::set<std::pair<Name, Name>>& contract_names,
            const std::set<Name>& fuse_names = {}) {
         using ResultScalarType = std::common_type_t<ScalarType1, ScalarType2>;
         using ResultTensor = Tensor<ResultScalarType, Symmetry, Name>;
         if constexpr (std::is_same_v<ResultScalarType, ScalarType1>) {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract_implement(tensor_1, tensor_2, contract_names, fuse_names);
            } else {
               return ResultTensor::contract_implement(tensor_1, tensor_2.template to<ResultScalarType>(), contract_names, fuse_names);
            }
         } else {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract_implement(tensor_1.template to<ResultScalarType>(), tensor_2, contract_names, fuse_names);
            } else {
               return ResultTensor::contract_implement(
                     tensor_1.template to<ResultScalarType>(),
                     tensor_2.template to<ResultScalarType>(),
                     contract_names,
                     fuse_names);
            }
         }
      }

      template<typename OtherScalarType, typename = std::enable_if_t<is_scalar<OtherScalarType>>>
      [[nodiscard]] auto contract(
            const Tensor<OtherScalarType, Symmetry, Name>& tensor_2,
            const std::set<std::pair<Name, Name>>& contract_names,
            const std::set<Name>& fuse_names = {}) const {
         return contract(*this, tensor_2, contract_names, fuse_names);
      }

      /**
       * Get the conjugated tensor
       * \note for symmetry tensor, every symmetry is transformed to -symmetry,
       * for fermion tensor, arrow is reversed, for complex tensor value got conjugated
       * \param positive_contract ensure the contract of result and self is positive.
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> conjugate(bool positive_contract = false) const;

      /**
       * Set the tensor as identity inplacely
       * \param pairs pair set describing how to treat the tensor as matrix
       */
      Tensor<ScalarType, Symmetry, Name>& identity(const std::set<std::pair<Name, Name>>& pairs) &;

      Tensor<ScalarType, Symmetry, Name>&& identity(const std::set<std::pair<Name, Name>>& pairs) && {
         return std::move(identity(pairs));
      }

      /**
       * Get the tensor exponential
       * \param pairs pair set describing how to treat the tensor as matrix
       * \param step iteration step
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> exponential(const std::set<std::pair<Name, Name>>& pairs, int step = 2) const;

      /**
       * Get trace of tensor
       * \param pairs pair set describing how to trace the tensor
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> trace(const std::set<std::pair<Name, Name>>& trace_names) const;

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
       * \param free_name_set_u U tensor free name after SVD
       * \param common_name_u U tensor new name after SVD
       * \param common_name_v V tensor new name after SVD
       * \param singular_name_u S tensor edge name connected to tensor U
       * \param singular_name_v S tensor edge name connected to tensor V
       * \param cut How to cut bond dimension during SVD
       * \return SVD result
       * \see svd_result
       */
      [[nodiscard]] svd_result
      svd(const std::set<Name>& free_name_set_u,
          const Name& common_name_u,
          const Name& common_name_v,
          const Name& singular_name_u,
          const Name& singular_name_v,
          Cut cut = NoCut()) const;

      /**
       * Calculate QR of the tensor
       * \param free_name_direction specify what tensor the free_name_set means, it can be 'Q' or 'R'
       * \param free_name_set one of tensor Q or tensor R free name after QR
       * \param common_name_q Q tensor new name after QR
       * \param common_name_r R tensor new name after QR
       * \return QR result
       * \see qr_result
       */
      [[nodiscard]] qr_result
      qr(char free_name_direction, const std::set<Name>& free_name_set, const Name& common_name_q, const Name& common_name_r) const;

      using EdgePointShrink = std::conditional_t<Symmetry::length == 0, Size, std::tuple<Symmetry, Size>>;
      using EdgePointExpand = std::conditional_t<
            Symmetry::length == 0,
            std::tuple<Size, Size>,
            std::conditional_t<Symmetry::is_fermi_symmetry, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;
      // index, dim

      /**
       * expand a dimension-1 edge of a tensor to several wider edge
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      expand(const std::unordered_map<Name, EdgePointExpand>& configure, const Name& old_name = InternalName<Name>::No_Old_Name) const;
      /**
       * shrink several edge of a tensor to a dimension-1 edge
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> shrink(
            const std::unordered_map<Name, EdgePointShrink>& configure,
            const Name& new_name = InternalName<Name>::No_New_Name,
            Arrow arrow = false) const;

      const Tensor<ScalarType, Symmetry, Name>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry, Name>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry, Name>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry, Name>& data_get(std::istream&);

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Tensor<ScalarType, Symmetry, Name>& load(const std::string&) &;
      Tensor<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };
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
         std::set<std::pair<Name, Name>> contract_names) {
      return tensor_1.contract(tensor_2, std::move(contract_names));
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   struct TensorShape {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>);

      const Tensor<ScalarType, Symmetry, Name>* owner;
   };

   // TODO quasi tensor (middle value between edge_operator)
} // namespace TAT
#endif
