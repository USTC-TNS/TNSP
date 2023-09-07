/**
 * \file edge.hpp
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
#ifndef TAT_EDGE_HPP
#define TAT_EDGE_HPP

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "symmetry.hpp"

namespace TAT {
    template<typename Symmetry, bool _is_pointer = false>
    class edge_segments_t {
        static_assert(is_symmetry<Symmetry>);

      public:
        static constexpr bool is_pointer = _is_pointer;
        static constexpr bool is_not_pointer = !is_pointer;

      public:
        using symmetry_t = Symmetry;
        using segments_t = std::vector<std::pair<symmetry_t, Size>>;

      private:
        std::conditional_t<is_pointer, const segments_t&, segments_t> m_segments;

      protected:
        using symlist_t = std::vector<symmetry_t>;

      private:
        static segments_t symlist_to_segments(const symlist_t& symmetries) {
            segments_t result;
            result.reserve(symmetries.size());
            for (const auto& symmetry : symmetries) {
                result.push_back({symmetry, 1});
            }
            return result;
        }

      public:
        edge_segments_t() = default;
        edge_segments_t(const edge_segments_t& edge) = default;
        edge_segments_t(edge_segments_t&& edge) = default;
        edge_segments_t& operator=(const edge_segments_t&) = default;
        edge_segments_t& operator=(edge_segments_t&&) = default;
        ~edge_segments_t() = default;

        // only valid if is_not_pointer
        edge_segments_t(segments_t&& segments) : m_segments(std::move(segments)) {
            static_assert(is_not_pointer);
        }
        // valid for both
        edge_segments_t(const segments_t& segments) : m_segments(segments) {
            static_assert(true);
        }

        /**
         * construct the edge with list of symmetry, each size of them are 1
         */
        edge_segments_t(const symlist_t& symmetries) : m_segments(symlist_to_segments(symmetries)) {
            static_assert(is_not_pointer);
        }

        /**
         * construct a trivial edge, only contain a single symmetry
         */
        edge_segments_t(const Size dimension, const symmetry_t symmetry = symmetry_t()) : m_segments({{symmetry, dimension}}) {
            static_assert(is_not_pointer);
        }

        // segment   = (symmetry, dimension)
        // segments  = [segment]
        // position  = index in segment list

        // index     = total index in the whole segments
        // offset    = index in single segment
        // point     = (symmetry, offset)
        // coord     = (position, offset)

      public:
        using index_t = Size;
        using position_t = Nums;
        using point_t = std::pair<symmetry_t, Size>;
        using coord_t = std::pair<position_t, Size>;

      public:
        const segments_t& segments() const {
            return m_segments;
        }
        position_t segments_size() const {
            return m_segments.size();
        }
        const std::pair<symmetry_t, Size>& segments(position_t i) const {
            return m_segments[i];
        }

      public:
        coord_t coord_by_point(const point_t& point) const {
            const auto& [symmetry, offset] = point;
            return {position_by_symmetry(symmetry), offset};
        }
        point_t point_by_coord(const coord_t& coord) const {
            const auto& [position, offset] = coord;
            return {segments(position).first, offset};
        }
        coord_t coord_by_index(const index_t& index) const {
            Size offset = index;
            for (auto position = 0; position < segments_size(); position++) {
                const auto& [symmetry, dimension] = segments(position);
                if (offset < dimension) {
                    return {position, offset};
                } else {
                    offset -= dimension;
                }
            }
            detail::error("Index is more than edge total dimension");
        }
        index_t index_by_coord(const coord_t& coord) const {
            return index_by_point(point_by_coord(coord));
        }
        point_t point_by_index(const index_t& index) const {
            return point_by_coord(coord_by_index(index));
        }
        index_t index_by_point(const point_t& point) const {
            Size result = point.second;
            for (const auto& [symmetry, dimension] : segments()) {
                if (symmetry == point.first) {
                    return result;
                }
                result += dimension;
            }
            detail::error("The symmetry not found in this edge");
        }

      public:
        auto find_by_symmetry(const symmetry_t& symmetry) const {
            return std::find_if(segments().begin(), segments().end(), [&symmetry](const auto& x) { return symmetry == x.first; });
        }

      public:
        position_t position_by_symmetry(const symmetry_t& symmetry) const {
            auto where = find_by_symmetry(symmetry);
            if constexpr (debug_mode) {
                if (where == segments().end()) {
                    detail::error("No such symmetry in this edge");
                }
            }
            return std::distance(segments().begin(), where);
        }

        Size dimension_by_symmetry(const symmetry_t& symmetry) const {
            auto where = find_by_symmetry(symmetry);
            if constexpr (debug_mode) {
                if (where == segments().end()) {
                    detail::error("No such symmetry in this edge");
                }
            }
            return where->second;
        }

        void conjugate() {
            static_assert(is_not_pointer);
            for (auto& [symmetry, dimension] : m_segments) {
                symmetry = -symmetry;
            }
        }

        edge_segments_t<symmetry_t> conjugated() const {
            segments_t result;
            result.reserve(segments_size());
            for (const auto& [symmetry, dimension] : segments()) {
                result.emplace_back(-symmetry, dimension);
            }
            return edge_segments_t<symmetry_t>(std::move(result));
        }

        Size total_dimension() const {
            Size total = 0;
            for (const auto& [symmetry, dimension] : segments()) {
                total += dimension;
            }
            return total;
        }
    };

    struct edge_bose_arrow_t {
        edge_bose_arrow_t() { }
        edge_bose_arrow_t(Arrow) { }

        static constexpr Arrow arrow() {
            return false;
        }
        static void set_arrow(Arrow) { }
        static void reverse_arrow() { }
    };

    // there are background EPR pair for each edge, for fermi edge, it is needed to record the order of this EPR pair, which is so called fermi arrow
    struct edge_fermi_arrow_t {
        edge_fermi_arrow_t() : m_arrow(false) { }
        edge_fermi_arrow_t(Arrow arrow) : m_arrow(arrow) { }

        Arrow arrow() const {
            return m_arrow;
        }
        void set_arrow(Arrow arrow) {
            m_arrow = arrow;
        }
        void reverse_arrow() {
            m_arrow ^= true;
        }
      private:
        Arrow m_arrow;
    };

    template<typename Symmetry>
    using edge_arrow_t = std::conditional_t<Symmetry::is_fermi_symmetry, edge_fermi_arrow_t, edge_bose_arrow_t>;

    /**
     * The shape of tensor edge, is a list of pair of symmetry and size, which construct a structure like line segment.
     * If it is fermi edge, an arrow is also included
     *
     * \tparam Symmetry The symmetry of the tensor
     * \tparam is_pointer whether it is just point to the data or the edge containing the real structure.
     */
    template<typename Symmetry, bool is_pointer = false>
    class Edge : public edge_segments_t<Symmetry, is_pointer>, public edge_arrow_t<Symmetry> {
        static_assert(is_symmetry<Symmetry>);

      public:
        using base_arrow_t = edge_arrow_t<Symmetry>;
        using base_segments_t = edge_segments_t<Symmetry, is_pointer>;

        Edge() = default;
        Edge(const Edge&) = default;
        Edge(Edge&&) = default;
        Edge& operator=(const Edge&) = default;
        Edge& operator=(Edge&&) = default;
        ~Edge() = default;

        template<
            typename Arg,
            typename =
                std::enable_if_t<std::is_constructible_v<base_segments_t, Arg&&> && !std::is_same_v<remove_cvref_t<Arg>, Edge<Symmetry, is_pointer>>>>
        Edge(Arg&& arg, Arrow arrow = false) : base_segments_t(std::forward<Arg>(arg)), base_arrow_t(arrow) { }
        Edge(std::initializer_list<std::pair<Symmetry, Size>> segments, Arrow arrow = false) :
            base_segments_t(typename base_segments_t::segments_t(segments)),
            base_arrow_t(arrow) { }
        Edge(std::initializer_list<Symmetry> symmetries, Arrow arrow = false) :
            base_segments_t(typename base_segments_t::symlist_t(symmetries)),
            base_arrow_t(arrow) { }

        void conjugate() {
            base_segments_t::conjugate();
            base_arrow_t::reverse_arrow();
        }
        Edge<Symmetry> conjugated() const {
            return Edge<Symmetry>(std::move(base_segments_t::conjugated()), !base_arrow_t::arrow());
        }
    };

    template<typename Symmetry, bool is_pointer>
    bool operator==(const edge_segments_t<Symmetry, is_pointer>& edge_1, const edge_segments_t<Symmetry, is_pointer>& edge_2) {
        return std::equal(edge_1.segments().begin(), edge_1.segments().end(), edge_2.segments().begin(), edge_2.segments().end());
    }
    template<typename Symmetry, bool is_pointer>
    bool operator!=(const edge_segments_t<Symmetry, is_pointer>& edge_1, const edge_segments_t<Symmetry, is_pointer>& edge_2) {
        return !(edge_1 == edge_2);
    }

    template<typename Symmetry, bool is_pointer>
    bool operator==(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
        return (edge_1.arrow() == edge_2.arrow()) && (edge_1.segments() == edge_2.segments());
    }
    template<typename Symmetry, bool is_pointer>
    bool operator!=(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
        return !(edge_1 == edge_2);
    }

    /**
     * An edge but only containing a pointer to other edge's segment data
     * \see Edge
     */
    template<typename Symmetry>
    using EdgePointer = Edge<Symmetry, true>;

    namespace detail {
        template<typename T>
        struct is_edge_helper : std::false_type { };

        template<typename T>
        struct is_edge_helper<Edge<T, false>> : std::true_type { };

        template<typename T>
        struct is_edge_pointer_helper : std::false_type { };

        template<typename T>
        struct is_edge_pointer_helper<Edge<T, true>> : std::true_type { };
    } // namespace detail

    template<typename T>
    constexpr bool is_edge = detail::is_edge_helper<T>::value;

    template<typename T>
    constexpr bool is_edge_pointer = detail::is_edge_pointer_helper<T>::value;

    template<typename T>
    constexpr bool is_general_edge = is_edge<T> || is_edge_pointer<T>;
} // namespace TAT
#endif
