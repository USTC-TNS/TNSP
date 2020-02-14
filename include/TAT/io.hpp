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

#include "tensor.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

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
      const auto pos = id_to_name.find(name.id);
      if (pos == id_to_name.end()) {
         return out << "UserDefinedName" << name.id;
      } else {
         return out << id_to_name.at(name.id);
      }
      return out;
   }

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec) {
      out << '[';
      auto not_first = false;
      for (const auto& i : vec) {
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
   template<class T>
   void raw_write_vector(std::ostream& out, const vector<T>& vec) {
      Size count = vec.size();
      raw_write(out, &count);
      raw_write(out, vec.data(), count);
   }
   template<class T>
   void raw_read_vector(std::istream& in, vector<T>& vec) {
      Size count;
      raw_read(in, &count);
      vec.resize(count);
      raw_read(in, vec.data(), count);
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
         for (const auto& [sym, dim] : edge.map) {
            if (not_first) {
               out << ',';
            }
            not_first = true;
            out << sym << ':' << dim;
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
         for (const auto& [sym, dim] : edge.map) {
            raw_write(out, &sym);
            raw_write(out, &dim);
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
            Symmetry sym;
            Size dim;
            raw_read(in, &sym);
            raw_read(in, &dim);
            edge.map[sym] = dim;
         }
      }
      return in;
   }

   std::ostream& operator<<(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s) {
      out << s.z2;
      return out;
   }
   std::ostream& operator<<(std::ostream& out, const U1Symmetry& s) {
      out << s.u1;
      return out;
   }
   std::ostream& operator<<(std::ostream& out, const FermiSymmetry& s) {
      out << s.fermi;
      return out;
   }
   std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& s) {
      out << '(' << s.fermi << ',' << s.z2 << ')';
      return out;
   }
   std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& s) {
      out << '(' << s.fermi << ',' << s.u1 << ')';
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
   inline const WindowsColorCode console_origin = {7};
#else
   inline const std::string console_red = "\x1B[31m";
   inline const std::string console_green = "\x1B[32m";
   inline const std::string console_yellow = "\x1B[33m";
   inline const std::string console_origin = "\x1B[0m";
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
         for (const auto& [i, j] : tensor.core->blocks) {
            if (not_first) {
               out << ',';
            }
            not_first = true;
            out << console_yellow << i << console_origin << ':' << j;
         }
         out << '}';
      }
      out << '}';
      return out;
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>&
   Tensor<ScalarType, Symmetry>::meta_put(std::ostream& out) const {
      raw_write_vector(out, names);
      for (const auto& edge : core->edges) {
         out <= edge;
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>&
   Tensor<ScalarType, Symmetry>::data_put(std::ostream& out) const {
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
      vector<Edge<Symmetry>> edges(rank);
      for (auto& edge : edges) {
         in >= edge;
      }
      core = std::make_shared<Core<ScalarType, Symmetry>>();
      core->edges = std::move(edges);
      if (!is_valid_name(names, core->edges.size())) {
         TAT_WARNING("Invalid Names");
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_get(std::istream& in) {
      Rank rank = names.size();
      Size count;
      raw_read(in, &count);
      core->blocks.clear();
      for (auto i = 0; i < count; i++) {
         auto symmetries = vector<Symmetry>(rank);
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
      try {
         std::clog << console_red << "\n\nPremature optimization is the root of all evil!\n"
                   << console_origin
                   << "                                       --- Donald Knuth\n\n\n";
      } catch ([[maybe_unused]] const std::exception& e) {
      }
   }
} // namespace TAT
#endif
