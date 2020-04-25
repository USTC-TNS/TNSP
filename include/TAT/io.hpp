/**
 * \file io.hpp
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

#pragma once
#ifndef TAT_IO_HPP
#define TAT_IO_HPP

#include <iostream>
#ifdef _WIN32
extern "C" {
#include <windows.h>
}
#endif

#include "tensor.hpp"

namespace TAT {
   template<class ScalarType>
   std::ostream& print_complex(std::ostream& out, const std::complex<ScalarType>& value) {
      if (value.real() != 0) {
         out << value.real();
         if (value.imag() != 0) {
            if (value.imag() > 0) {
               out << '+';
            }
            out << value.imag();
            out << 'i';
         }
      } else {
         if (value.imag() == 0) {
            out << '0';
         } else {
            out << value.imag();
            out << 'i';
         }
      }
      return out;
   }

   template<class T>
   void raw_write(std::ostream& out, const T* data, const Size number = 1) {
      out.write(reinterpret_cast<const char*>(data), sizeof(T) * number);
   }
   template<class T>
   void raw_read(std::istream& in, T* data, const Size number = 1) {
      in.read(reinterpret_cast<char*>(data), sizeof(T) * number);
   }

   inline std::ostream& operator<<(std::ostream& out, const Name& name) {
      if (const auto position = id_to_name.find(name.id); position == id_to_name.end()) {
         return out << "UserDefinedName" << name.id;
      } else {
         return out << position->second;
      }
   }

   template<class T, class A>
   std::ostream& operator<<(std::ostream& out, const std::vector<T, A>& list) {
      out << '[';
      auto not_first = false;
      for (const auto& i : list) {
         if (not_first) {
            out << ',';
         }
         not_first = true;
         if constexpr (std::is_same_v<T, std::complex<real_base_t<T>>>) {
            print_complex(out, i);
         } else {
            out << i;
         }
      }
      out << ']';
      return out;
   }
   template<class T, class A>
   void raw_write_vector(std::ostream& out, const std::vector<T, A>& list) {
      Size count = list.size();
      raw_write(out, &count);
      raw_write(out, list.data(), count);
   }
   template<class T, class A>
   void raw_read_vector(std::istream& in, std::vector<T, A>& list) {
      Size count;
      raw_read(in, &count);
      list.resize(count);
      raw_read(in, list.data(), count);
   }
   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << edge.map.at(NoSymmetry());
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out << '{' << "arrow" << ':';
            out << edge.arrow;
            out << ',' << "map" << ':';
         }
         out << '{';
         auto not_first = false;
         for (const auto& [symmetry, dimension] : edge.map) {
            if (not_first) {
               out << ',';
            }
            not_first = true;
            out << symmetry << ':' << dimension;
         }
         out << '}';
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out << '}';
         }
      }
      return out;
   }
   template<class Symmetry>
   std::ostream& operator<=(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         raw_write(out, &edge.map.begin()->second);
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            raw_write(out, &edge.arrow);
         }
         const Nums numbers = edge.map.size();
         raw_write(out, &numbers);
         for (const auto& [symmetry, dimension] : edge.map) {
            raw_write(out, &symmetry);
            raw_write(out, &dimension);
         }
      }
      return out;
   }
   template<class Symmetry>
   std::istream& operator>=(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         Size dim;
         raw_read(in, &dim);
         edge.map[NoSymmetry()] = dim;
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            raw_read(in, &edge.arrow);
         }
         Nums numbers;
         raw_read(in, &numbers);
         edge.map.clear();
         for (Nums i = 0; i < numbers; i++) {
            Symmetry symmetry;
            Size dimension;
            raw_read(in, &symmetry);
            raw_read(in, &dimension);
            edge.map[symmetry] = dimension;
         }
      }
      return in;
   }

   inline std::ostream& operator<<(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   inline std::ostream& operator<<(std::ostream& out, const Z2Symmetry& symmetry) {
      out << symmetry.z2;
      return out;
   }
   inline std::ostream& operator<<(std::ostream& out, const U1Symmetry& symmetry) {
      out << symmetry.u1;
      return out;
   }
   inline std::ostream& operator<<(std::ostream& out, const FermiSymmetry& symmetry) {
      out << symmetry.fermi;
      return out;
   }
   inline std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& symmetry) {
      out << '(' << symmetry.fermi << ',' << symmetry.z2 << ')';
      return out;
   }
   inline std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& symmetry) {
      out << '(' << symmetry.fermi << ',' << symmetry.u1 << ')';
      return out;
   }

#ifdef _WIN32
   inline const auto stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
   inline const auto stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
   struct WindowsColorCode {
      int color_code;
   };
   inline std::ostream& operator<<(std::ostream& out, const WindowsColorCode& value) {
      if (out.rdbuf() == std::cout.rdbuf()) {
         SetConsoleTextAttribute(stdout_handle, value.color_code);
      } else if (out.rdbuf() == std::clog.rdbuf() || out.rdbuf() == std::cerr.rdbuf()) {
         SetConsoleTextAttribute(stderr_handle, value.color_code);
      }
      return out;
   }
   inline const WindowsColorCode console_red = {4};
   inline const WindowsColorCode console_green = {2};
   inline const WindowsColorCode console_yellow = {6};
   inline const WindowsColorCode console_blue = {1};
   inline const WindowsColorCode console_origin = {7};
#else
   struct UnixColorCode {
      std::string color_code;
   };
   inline const UnixColorCode console_red = {"\x1B[31m"};
   inline const UnixColorCode console_green = {"\x1B[32m"};
   inline const UnixColorCode console_yellow = {"\x1B[33m"};
   inline const UnixColorCode console_blue = {"\x1B[34m"};
   inline const UnixColorCode console_origin = {"\x1B[0m"};
   inline std::ostream& operator<<(std::ostream& out, const UnixColorCode& value) {
#ifdef TAT_ALWAYS_COLOR
      out << value.color_code;
#else
      if (out.rdbuf() == std::cout.rdbuf() || out.rdbuf() == std::clog.rdbuf() || out.rdbuf() == std::cerr.rdbuf()) {
         out << value.color_code;
      }
#endif
      return out;
   }
#endif

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      out << '{' << console_green << "names" << console_origin << ':';
      out << tensor.names;
      out << ',' << console_green << "edges" << console_origin << ':';
      out << tensor.core->edges;
      out << ',' << console_green << "blocks" << console_origin << ':';
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << tensor.core->blocks.begin()->second;
      } else {
         out << '{';
         auto not_first = false;
         for (const auto& [symmetries, block] : tensor.core->blocks) {
            if (not_first) {
               out << ',';
            }
            not_first = true;
            out << console_yellow << symmetries << console_origin << ':' << block;
         }
         out << '}';
      }
      out << '}';
      return out;
   }

   template<class ScalarType, class Symmetry>
   std::string Tensor<ScalarType, Symmetry>::show() const {
      std::stringstream out;
      out << *this;
      return out.str();
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::meta_put(std::ostream& out) const {
      raw_write_vector(out, names);
      for (const auto& edge : core->edges) {
         out <= edge;
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_put(std::ostream& out) const {
      Size count = core->blocks.size();
      raw_write(out, &count);
      for (const auto& [i, j] : core->blocks) {
         raw_write(out, i.data(), i.size());
         raw_write_vector(out, j);
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   std::ostream& operator<=(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      tensor.meta_put(out).data_put(out);
      return out;
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::meta_get(std::istream& in) {
      raw_read_vector(in, names);
      const Rank rank = names.size();
      name_to_index = construct_name_to_index(names);
      std::vector<Edge<Symmetry>> edges(rank);
      for (auto& edge : edges) {
         in >= edge;
      }
      core = std::make_shared<Core<ScalarType, Symmetry>>();
      core->edges = std::move(edges);
      check_valid_name(names, core->edges.size());
      return *this;
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_get(std::istream& in) {
      Rank rank = names.size();
      Size count;
      raw_read(in, &count);
      core->blocks.clear();
      for (auto i = 0; i < count; i++) {
         auto symmetries = std::vector<Symmetry>(rank);
         raw_read(in, symmetries.data(), symmetries.size());
         raw_read_vector(in, core->blocks[std::move(symmetries)]);
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   std::istream& operator>=(std::istream& in, Tensor<ScalarType, Symmetry>& tensor) {
      tensor.meta_get(in).data_get(in);
      return in;
   }

   template<class T>
   std::istream&& operator>=(std::istream&& in, T& v) {
      in >= v;
      return std::move(in);
   }
   template<class T>
   std::ostream&& operator<=(std::ostream&& out, const T& v) {
      out <= v;
      return std::move(out);
   }

   inline Evil::~Evil() {
#ifndef NDEUBG
      try {
         std::clog << console_blue << "\n\nPremature optimization is the root of all evil!\n"
                   << console_origin << "                                       --- Donald Knuth\n\n\n";
      } catch ([[maybe_unused]] const std::exception& e) {
      }
#endif
   }

   inline void warning_or_error([[maybe_unused]] const std::string& message) {
#ifndef NDEBUG
      std::cerr << console_red << message << console_origin << std::endl;
#endif
   }
} // namespace TAT
#endif
