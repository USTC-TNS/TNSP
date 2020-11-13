/**
 * \file text_io.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_TEXT_IO_HPP
#define TAT_TEXT_IO_HPP

#include <iostream>
#include <limits>

#include "tensor.hpp"

namespace TAT {
   template<typename ScalarType, typename Symmetry>
   void text_out(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      out << "Tensor with rank = " << tensor.names.size() << "\n";
      out << "  Names = ";
      for (const auto& i : tensor.names) {
         out << i << " ";
      }
      out << "\n";
      out << "  Edges =\n";
      for (const auto& i : tensor.core->edges) {
         out << "    ";
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out << "Arrow = " << i.arrow << ", ";
         }
         out << "Map size = " << i.map.size() << ", Map = ";
         for (const auto& [key, value] : i.map) {
            out << key << "=>" << value << " ";
         }
         out << "\n";
      }
      out << "  Block number = " << tensor.core->blocks.size() << ", Blocks =\n";
      for (const auto& [key, value] : tensor.core->blocks) {
         out << "    [";
         bool not_first = false;
         for (const auto& s : key) {
            if (not_first) {
               out << ", ";
            }
            not_first = true;
            out << s;
         }
         out << "] : ";
         for (const auto& s : value) {
            out << s << " ";
         }
         out << "\n";
      }
   }

   Name read_name_from_text(std::istream& in) {
      char buffer[256]; // max name length = 256
      Size length = 0;
      while (in.peek() != ' ') {
         buffer[length++] = in.get();
      }
      buffer[length] = '\x00';
      return Name((const char*)buffer);
   }

   template<typename Symmetry>
   Symmetry read_symmetry_from_text(std::istream& in);

   template<>
   NoSymmetry read_symmetry_from_text<NoSymmetry>(std::istream& in) {
      return NoSymmetry();
   }
   template<>
   Z2Symmetry read_symmetry_from_text<Z2Symmetry>(std::istream& in) {
      auto result = Z2Symmetry();
      in >> result.z2;
      return result;
   }
   template<>
   U1Symmetry read_symmetry_from_text<U1Symmetry>(std::istream& in) {
      auto result = U1Symmetry();
      in >> result.u1;
      return result;
   }
   template<>
   FermiSymmetry read_symmetry_from_text<FermiSymmetry>(std::istream& in) {
      auto result = FermiSymmetry();
      in >> result.fermi;
      return result;
   }
   template<>
   FermiZ2Symmetry read_symmetry_from_text<FermiZ2Symmetry>(std::istream& in) {
      auto result = FermiZ2Symmetry();
      in.get();
      in >> result.fermi >> result.z2;
      in.get();
      return result;
   }
   template<>
   FermiU1Symmetry read_symmetry_from_text<FermiU1Symmetry>(std::istream& in) {
      auto result = FermiU1Symmetry();
      in.get();
      in >> result.fermi >> result.u1;
      in.get();
      return result;
   }

   template<typename ScalarType, typename Symmetry>
   void text_in(std::stringstream& in, Tensor<ScalarType, Symmetry>& tensor) {
      constexpr auto max_stream_size = std::numeric_limits<std::streamsize>::max();
      in.ignore(max_stream_size, '=');
      Rank rank;
      in >> rank;
      in.ignore(max_stream_size, '=');
      auto name_list = std::vector<Name>();
      name_list.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         in.get(); // 从"="开始每个name前面只有一个空格
         name_list.push_back(read_name_from_text(in));
      }
      in.ignore(max_stream_size, '=');
      auto edge_list = std::vector<Edge<Symmetry>>();
      edge_list.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto& this_edge = edge_list.emplace_back();
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            in.ignore(max_stream_size, '=');
            in >> this_edge.arrow;
         }
         in.ignore(max_stream_size, '=');
         Size map_size;
         in >> map_size;
         in.ignore(max_stream_size, '=');
         for (auto j = 0; j < map_size; j++) {
            auto& this_dimension = this_edge.map[read_symmetry_from_text<Symmetry>(in)];
            in.ignore(max_stream_size, '>');
            in >> this_dimension;
            in.get(); // 对称性难以定位，需要自己把空格去掉
         }
      }
      tensor = Tensor<ScalarType, Symmetry>(std::move(name_list), std::move(edge_list));
      in.ignore(max_stream_size, '=');
      Size block_number;
      in >> block_number;
      in.ignore(max_stream_size, '=');
      for (auto i = 0; i < block_number; i++) {
         in.ignore(max_stream_size, '[');
         auto symmetry = std::vector<Symmetry>();
         for (auto j = 0; j < rank; j++) {
            symmetry.push_back(read_symmetry_from_text<Symmetry>(in));
            in.get();
         }
         in.ignore(max_stream_size, ':');
         auto& block = tensor.core->blocks.at(symmetry);
         for (auto& value : block) {
            in >> value;
         }
      }
   }
} // namespace TAT

#endif
