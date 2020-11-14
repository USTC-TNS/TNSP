/**
 * \file io.hpp
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
#ifndef TAT_IO_HPP
#define TAT_IO_HPP

#include <iostream>
#include <limits>

#include "tensor.hpp"

namespace TAT {
   /**
    * 简洁地打印复数
    */
   template<typename ScalarType>
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

   template<typename ScalarType>
   std::ostream&& print_complex(std::ostream&& out, const std::complex<ScalarType>& value) {
      print_complex(out, value);
      return std::move(out);
   }

   inline void ignore_util(std::istream& in, char end) {
      in.ignore(std::numeric_limits<std::streamsize>::max(), end);
   }

   template<typename ScalarType>
   std::istream& scan_complex(std::istream& in, std::complex<ScalarType>& value) {
      ScalarType part;
      in >> part;
      char maybe_i = in.peek();
      if (maybe_i == 'i') {
         in.get();
         // no real part
         value = std::complex<ScalarType>{0, part};
      } else {
         // have real part
         if (maybe_i == '+' || maybe_i == '-') {
            // have imag part
            ScalarType another_part;
            in >> another_part;
            value = std::complex<ScalarType>{part, another_part};
            if (in.get() != 'i') {
               in.setstate(std::ios::failbit);
            }
         } else {
            // no imag part
            value = std::complex<ScalarType>{part, 0};
         }
      }
      return in;
   }

   template<typename ScalarType>
   std::istream&& scan_complex(std::istream&& in, std::complex<ScalarType>& value) {
      scan_complex(in, value);
      return std::move(in);
   }

   template<typename T, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   void raw_write(std::ostream& out, const T* data, Size number = 1) {
      out.write(reinterpret_cast<const char*>(data), sizeof(T) * number);
   }
   template<typename T, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   void raw_read(std::istream& in, T* data, Size number = 1) {
      in.read(reinterpret_cast<char*>(data), sizeof(T) * number);
   }

   inline std::ostream& operator<<(std::ostream& out, const Name& name) {
#ifdef TAT_USE_SIMPLE_NAME
      return out << name.name;
#else
      if (const auto position = id_to_name.find(name.id); position == id_to_name.end()) {
         return out << "UserDefinedName" << name.id;
      } else {
         return out << position->second;
      }
#endif
   }

   bool valid_name_character(char c) {
      return ' ' < c && c < '\x7f' && c != ',' && c != '[' && c != ']';
      // 可打印字符去掉空格，逗号和方括号
   }

   inline std::istream& operator>>(std::istream& in, Name& name) {
      char buffer[256]; // max name length = 256
      Size length = 0;
      while (valid_name_character(in.peek())) {
         buffer[length++] = in.get();
      }
      buffer[length] = '\x00';
      name = Name((const char*)buffer);
      return in;
   }

   template<typename T, typename A, typename = std::enable_if_t<is_scalar_v<T> || std::is_same_v<T, Name> || is_edge_v<T> || is_symmetry_v<T>>>
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

   template<typename T, typename A, typename = std::enable_if_t<is_scalar_v<T> || std::is_same_v<T, Name> || is_edge_v<T> || is_symmetry_v<T>>>
   std::istream& operator>>(std::istream& in, std::vector<T, A>& list) {
      ignore_util(in, '[');
      list.clear();
      if (in.peek() == ']') {
         // empty list
         in.get(); // 获取']'
      } else {
         // not empty
         while (true) {
            // 此时没有space
            auto& i = list.emplace_back();
            if constexpr (std::is_same_v<T, std::complex<real_base_t<T>>>) {
               scan_complex(in, i);
            } else {
               in >> i;
            }
            char next = in.get();
            if (next == ']') {
               break;
            }
         }
      }
      return in;
   }

   template<typename T, typename A, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   void raw_write_vector(std::ostream& out, const std::vector<T, A>& list) {
      Size count = list.size();
      raw_write(out, &count);
      raw_write(out, list.data(), count);
   }
   template<typename T, typename A, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   void raw_read_vector(std::istream& in, std::vector<T, A>& list) {
      Size count;
      raw_read(in, &count);
      list.resize(count);
      raw_read(in, list.data(), count);
   }

   void raw_write_string(std::ostream& out, const std::string& string) {
      Size count = string.size();
      raw_write(out, &count);
      raw_write(out, string.data(), count);
   }
   void raw_read_string(std::istream& in, std::string& string) {
      Size count;
      raw_read(in, &count);
      string.resize(count);
      raw_read(in, string.data(), count);
   }

   template<typename Symmetry>
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
   template<typename Symmetry>
   std::istream& operator>>(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         in >> edge.map[NoSymmetry()];
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            ignore_util(in, ':');
            in >> edge.arrow;
         }
         ignore_util(in, '{');
         edge.map.clear();
         if (in.peek() != '}') {
            // not empty
            do {
               Symmetry symmetry;
               in >> symmetry;
               ignore_util(in, ':');
               Size dimension;
               in >> dimension;
               edge.map[symmetry] = dimension;
            } while (in.get() == ','); // 读了map最后的'}'
         } else {
            in.get(); // 读了map最后的'}'
         }
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            ignore_util(in, '}');
         }
      }
      return in;
   }

   template<typename Symmetry>
   void raw_write_edge(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         raw_write(out, &edge.map.begin()->second);
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            raw_write(out, &edge.arrow);
         }
         const Rank numbers = edge.map.size();
         raw_write(out, &numbers);
         for (const auto& [symmetry, dimension] : edge.map) {
            raw_write(out, &symmetry);
            raw_write(out, &dimension);
         }
      }
   }
   template<typename Symmetry>
   void raw_read_edge(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         Size dim;
         raw_read(in, &dim);
         edge.map[NoSymmetry()] = dim;
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            raw_read(in, &edge.arrow);
         }
         Rank numbers;
         raw_read(in, &numbers);
         edge.map.clear();
         for (auto i = 0; i < numbers; i++) {
            Symmetry symmetry;
            Size dimension;
            raw_read(in, &symmetry);
            raw_read(in, &dimension);
            edge.map[symmetry] = dimension;
         }
      }
   }

   inline std::ostream& operator<<(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   inline std::istream& operator>>(std::istream& in, NoSymmetry&) {
      return in;
   }
   inline std::ostream& operator<<(std::ostream& out, const Z2Symmetry& symmetry) {
      out << symmetry.z2;
      return out;
   }
   inline std::istream& operator>>(std::istream& in, Z2Symmetry& symmetry) {
      in >> symmetry.z2;
      return in;
   }
   inline std::ostream& operator<<(std::ostream& out, const U1Symmetry& symmetry) {
      out << symmetry.u1;
      return out;
   }
   inline std::istream& operator>>(std::istream& in, U1Symmetry& symmetry) {
      in >> symmetry.u1;
      return in;
   }
   inline std::ostream& operator<<(std::ostream& out, const FermiSymmetry& symmetry) {
      out << symmetry.fermi;
      return out;
   }
   inline std::istream& operator>>(std::istream& in, FermiSymmetry& symmetry) {
      in >> symmetry.fermi;
      return in;
   }
   inline std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& symmetry) {
      out << '(' << symmetry.fermi << ',' << symmetry.z2 << ')';
      return out;
   }
   inline std::istream& operator>>(std::istream& in, FermiZ2Symmetry& symmetry) {
      ignore_util(in, '(');
      in >> symmetry.fermi;
      ignore_util(in, ',');
      in >> symmetry.z2;
      ignore_util(in, ')');
      return in;
   }

   inline std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& symmetry) {
      out << '(' << symmetry.fermi << ',' << symmetry.u1 << ')';
      return out;
   }
   inline std::istream& operator>>(std::istream& in, FermiU1Symmetry& symmetry) {
      ignore_util(in, '(');
      in >> symmetry.fermi;
      ignore_util(in, ',');
      in >> symmetry.u1;
      ignore_util(in, ')');
      return in;
   }

   /**
    * \brief 一个控制屏幕字体色彩的简单类型
    */
   struct UnixColorCode {
      std::string color_code;
      UnixColorCode(const char* code) noexcept : color_code(code) {}
   };
   inline const UnixColorCode console_red = "\x1B[31m";
   inline const UnixColorCode console_green = "\x1B[32m";
   inline const UnixColorCode console_yellow = "\x1B[33m";
   inline const UnixColorCode console_blue = "\x1B[34m";
   inline const UnixColorCode console_origin = "\x1B[0m";
   inline std::ostream& operator<<(std::ostream& out, const UnixColorCode& value) {
      out << value.color_code;
      return out;
   }

   template<typename ScalarType, typename Symmetry>
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
   template<typename ScalarType, typename Symmetry>
   std::istream& operator>>(std::istream& in, Tensor<ScalarType, Symmetry>& tensor) {
      ignore_util(in, ':');
      in >> tensor.names;
      tensor.name_to_index = construct_name_to_index(tensor.names);
      ignore_util(in, ':');
      tensor.core = std::make_shared<Core<ScalarType, Symmetry>>();
      in >> tensor.core->edges;
      check_valid_name(tensor.names, tensor.core->edges.size());
      ignore_util(in, ':');
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         // change begin();
         in >> tensor.core->blocks[std::vector<Symmetry>(tensor.names.size(), Symmetry())];
      } else {
         ignore_util(in, '{');
         if (in.peek() != '}') {
            do {
               std::vector<Symmetry> symmetries;
               in >> symmetries;
               ignore_util(in, ':');
               auto& data = tensor.core->blocks[std::move(symmetries)];
               in >> data;
            } while (in.get() == ','); // 读了map最后的'}'
         } else {
            in.get(); // 读了map最后的'}'
         }
      }
      ignore_util(in, '}');
      return in;
   }

   template<typename ScalarType, typename Symmetry>
   std::ostream& operator<<(std::ostream& out, const Singular<ScalarType, Symmetry>& singular) {
      const auto& value = singular.value; // std::map<Symmetry, vector<real_base_t<ScalarType>>>
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << value.begin()->second;
      } else {
         out << '{';
         bool first = true;
         for (const auto& [key, value] : value) {
            if (!first) {
               out << ',';
            } else {
               first = false;
            }
            out << console_yellow << key << console_origin << ':' << value;
         }
         out << '}';
      }
      return out;
   }

   template<typename ScalarType, typename Symmetry>
   std::string Tensor<ScalarType, Symmetry>::show() const {
      std::stringstream out;
      out << *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry>
   std::string Singular<ScalarType, Symmetry>::show() const {
      std::stringstream out;
      out << *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry>
   const Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::meta_put(std::ostream& out) const {
#ifdef TAT_USE_SIMPLE_NAME
      Rank count = names.size();
      raw_write(out, &count);
      for (const auto& name : names) {
         raw_write_string(out, name.name);
      }
#else
      raw_write_vector(out, names);
#endif
      for (const auto& edge : core->edges) {
         raw_write_edge(out, edge);
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry>
   const Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_put(std::ostream& out) const {
      Size count = core->blocks.size();
      raw_write(out, &count);
      for (const auto& [i, j] : core->blocks) {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         } else {
            raw_write(out, i.data(), i.size());
         }
         raw_write_vector(out, j);
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry>
   std::ostream& operator<(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      tensor.meta_put(out).data_put(out);
      return out;
   }

   template<typename ScalarType, typename Symmetry>
   std::string Tensor<ScalarType, Symmetry>::dump() const {
      std::stringstream out;
      out < *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry>
   std::ostream& operator<(std::ostream& out, const Singular<ScalarType, Symmetry>& singular) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         raw_write_vector(out, singular.value.begin()->second);
      } else {
         const Rank numbers = singular.value.size();
         raw_write(out, &numbers);
         for (const auto& [symmetry, vector] : singular.value) {
            raw_write(out, &symmetry);
            raw_write_vector(out, vector);
         }
      }
      return out;
   }

   template<typename ScalarType, typename Symmetry>
   std::string Singular<ScalarType, Symmetry>::dump() const {
      std::stringstream out;
      out < *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::meta_get(std::istream& in) {
#ifdef TAT_USE_SIMPLE_NAME
      Rank count;
      raw_read(in, &count);
      names.clear();
      names.reserve(count);
      for (auto i = 0; i < count; i++) {
         std::string this_name;
         raw_read_string(in, this_name);
         names.push_back(std::move(this_name));
      }
#else
      raw_read_vector(in, names);
#endif
      const Rank rank = names.size();
      name_to_index = construct_name_to_index(names);
      core = std::make_shared<Core<ScalarType, Symmetry>>();
      core->edges.clear();
      core->edges.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto& edge = core->edges.emplace_back();
         raw_read_edge(in, edge);
      }
      check_valid_name(names, core->edges.size());
      return *this;
   }

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_get(std::istream& in) {
      Rank rank = names.size();
      Size count;
      raw_read(in, &count);
      core->blocks.clear();
      for (auto i = 0; i < count; i++) {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            raw_read_vector(in, core->blocks[std::vector<NoSymmetry>(rank, NoSymmetry())]);
         } else {
            auto symmetries = std::vector<Symmetry>(rank);
            raw_read(in, symmetries.data(), symmetries.size());
            raw_read_vector(in, core->blocks[std::move(symmetries)]);
         }
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry>
   std::istream& operator>(std::istream& in, Tensor<ScalarType, Symmetry>& tensor) {
      tensor.meta_get(in).data_get(in);
      return in;
   }

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::load(const std::string& input) & {
      std::stringstream in(input);
      in > *this;
      return *this;
   }

   template<typename ScalarType, typename Symmetry>
   std::istream& operator>(std::istream& in, Singular<ScalarType, Symmetry>& singular) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         vector<real_base_t<ScalarType>> singulars;
         raw_read_vector(in, singulars);
         singular.value[NoSymmetry()] = std::move(singulars);
      } else {
         Rank numbers;
         raw_read(in, &numbers);
         singular.value.clear();
         for (auto i = 0; i < numbers; i++) {
            Symmetry symmetry;
            vector<real_base_t<ScalarType>> singulars;
            raw_read(in, &symmetry);
            raw_read_vector(in, singulars);
            singular.value[symmetry] = std::move(singulars);
         }
      }
      return in;
   }

   template<typename ScalarType, typename Symmetry>
   Singular<ScalarType, Symmetry>& Singular<ScalarType, Symmetry>::load(const std::string& input) & {
      std::stringstream in(input);
      in > *this;
      return *this;
   }

   template<typename T>
   std::istream&& operator>(std::istream&& in, T& v) {
      in > v;
      return std::move(in);
   }
   template<typename T>
   std::ostream&& operator<(std::ostream&& out, const T& v) {
      out < v;
      return std::move(out);
   }
} // namespace TAT
#endif
