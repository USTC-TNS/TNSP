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
   std::ostream& operator<(std::ostream& out, const T& data) {
      out.write(reinterpret_cast<const char*>(&data), sizeof(T));
      return out;
   }
   template<typename T, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   std::istream& operator>(std::istream& in, T& data) {
      in.read(reinterpret_cast<char*>(&data), sizeof(T));
      return in;
   }

   // 如果Name = std::string则不能使用这个来输出
   // 而输入的话会重载std::string的输入问题不大
   // 对于二进制io在tensor处处理了问题也不大
   inline std::ostream& operator<<(std::ostream& out, const FastName& name) {
      if (const auto position = fast_name_dataset.id_to_name.find(name.id); position == fast_name_dataset.id_to_name.end()) {
         return out << "UserDefinedName" << name.id;
      } else {
         return out << position->second;
      }
   }

   bool valid_name_character(char c) {
      return ' ' < c && c < '\x7f' && c != ',' && c != '[' && c != ']';
      // 可打印字符去掉空格，逗号和方括号
   }

   // inline std::istream& operator>>(std::istream& in, std::string& name) {
   inline std::string read_string_for_name(std::istream& in) {
      char buffer[256]; // max name length = 256
      Size length = 0;
      while (valid_name_character(in.peek())) {
         buffer[length++] = in.get();
      }
      buffer[length] = '\x00';
      return std::string((const char*)buffer);
   }

   inline std::istream& operator>>(std::istream& in, FastName& name) {
      name = FastName(read_string_for_name(in));
      return in;
   }

   std::ostream& operator<(std::ostream& out, const std::string& string) {
      Size count = string.size();
      out < count;
      out.write(string.data(), sizeof(char) * count);
      return out;
   }
   std::istream& operator>(std::istream& in, std::string& string) {
      Size count;
      in > count;
      string.resize(count);
      in.read(string.data(), sizeof(char) * count);
      return in;
   }

   template<typename Key, typename Value>
   std::ostream& operator<(std::ostream& out, const std::map<Key, Value>& map) {
      Size size = map.size();
      out < size;
      for (const auto& [key, value] : map) {
         out < key < value;
      }
      return out;
   }

   template<typename Key, typename Value>
   std::istream& operator>(std::istream& in, std::map<Key, Value>& map) {
      Size size;
      in > size;
      map.clear();
      for (auto i = 0; i < size; i++) {
         Key key;
         in > key;
         in > map[std::move(key)];
      }
      return in;
   }

   template<
         typename T,
         typename A,
         typename =
               std::enable_if_t<is_scalar_v<T> || std::is_same_v<T, std::string> || std::is_same_v<T, FastName> || is_edge_v<T> || is_symmetry_v<T>>>
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

   template<
         typename T,
         typename A,
         typename =
               std::enable_if_t<is_scalar_v<T> || std::is_same_v<T, std::string> || std::is_same_v<T, FastName> || is_edge_v<T> || is_symmetry_v<T>>>
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
            if constexpr (std::is_same_v<T, std::string>) {
               i = read_string_for_name(in);
            }
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

   template<typename T, typename A>
   std::ostream& operator<(std::ostream& out, const std::vector<T, A>& list) {
      Size count = list.size();
      out < count;
      if constexpr (std::is_trivially_destructible_v<T>) {
         out.write(reinterpret_cast<const char*>(list.data()), sizeof(T) * count);
      } else {
         for (const auto& i : list) {
            out < i;
         }
      }
      return out;
   }
   template<typename T, typename A>
   std::istream& operator>(std::istream& in, std::vector<T, A>& list) {
      Size count;
      in > count;
      if constexpr (std::is_trivially_destructible_v<T>) {
         list.resize(count);
         in.read(reinterpret_cast<char*>(list.data()), sizeof(T) * count);
      } else {
         for (auto i = 0; i < count; i++) {
            auto& item = list.emplace_back();
            in > item;
         }
      }
      return in;
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
   std::ostream& operator<(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out < edge.map.begin()->second;
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out < edge.arrow;
         }
         out < edge.map;
      }
      return out;
   }
   template<typename Symmetry>
   std::istream& operator>(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         // edge.map.clear(); 不需要
         in > edge.map[NoSymmetry()];
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            in > edge.arrow;
         }
         in > edge.map;
      }
      return in;
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

   template<typename ScalarType, typename Symmetry, typename Name>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry, Name>& tensor) {
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
   template<typename ScalarType, typename Symmetry, typename Name>
   std::istream& operator>>(std::istream& in, Tensor<ScalarType, Symmetry, Name>& tensor) {
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

   template<typename ScalarType, typename Symmetry, typename Name>
   std::ostream& operator<<(std::ostream& out, const Singular<ScalarType, Symmetry, Name>& singular) {
      const auto& value = singular.value; // std::map<Symmetry, vector<real_base_t<ScalarType>>>
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << value.begin()->second;
      } else {
         out << '{';
         bool not_first = false;
         for (const auto& [key, value] : value) {
            if (not_first) {
               out << ',';
            } else {
               not_first = true;
            }
            out << console_yellow << key << console_origin << ':' << value;
         }
         out << '}';
      }
      return out;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Tensor<ScalarType, Symmetry, Name>::show() const {
      std::stringstream out;
      out << *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Singular<ScalarType, Symmetry, Name>::show() const {
      std::stringstream out;
      out << *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   const Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::meta_put(std::ostream& out) const {
      out < names;
      for (const auto& edge : core->edges) {
         out < edge;
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   const Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::data_put(std::ostream& out) const {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out < core->blocks.begin()->second;
      } else {
         out < core->blocks;
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::ostream& operator<(std::ostream& out, const Tensor<ScalarType, Symmetry, Name>& tensor) {
      tensor.meta_put(out).data_put(out);
      return out;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Tensor<ScalarType, Symmetry, Name>::dump() const {
      std::stringstream out;
      out < *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::ostream& operator<(std::ostream& out, const Singular<ScalarType, Symmetry, Name>& singular) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out < singular.value.begin()->second;
      } else {
         out < singular.value;
      }
      return out;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Singular<ScalarType, Symmetry, Name>::dump() const {
      std::stringstream out;
      out < *this;
      return out.str();
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::meta_get(std::istream& in) {
      in > names;
      const Rank rank = names.size();
      name_to_index = construct_name_to_index(names);
      core = std::make_shared<Core<ScalarType, Symmetry>>();
      core->edges.clear();
      core->edges.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto& edge = core->edges.emplace_back();
         in > edge;
      }
      check_valid_name(names, core->edges.size());
      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::data_get(std::istream& in) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         core->blocks.clear();
         in > core->blocks[std::vector<NoSymmetry>(names.size(), NoSymmetry())];
      } else {
         in > core->blocks;
      }
      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::istream& operator>(std::istream& in, Tensor<ScalarType, Symmetry, Name>& tensor) {
      tensor.meta_get(in).data_get(in);
      return in;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::load(const std::string& input) & {
      std::stringstream in(input);
      in > *this;
      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::istream& operator>(std::istream& in, Singular<ScalarType, Symmetry, Name>& singular) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         // singular.value.clear(); 不需要
         in > singular.value[NoSymmetry()];
      } else {
         in > singular.value;
      }
      return in;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Singular<ScalarType, Symmetry, Name>& Singular<ScalarType, Symmetry, Name>::load(const std::string& input) & {
      std::stringstream in(input);
      in > *this;
      return *this;
   }

   inline std::ostream& operator<(std::ostream& out, const fast_name_dataset_t& dataset) {
      return out < dataset.names_total_index < dataset.name_to_id < dataset.id_to_name;
   }
   inline std::istream& operator>(std::istream& in, fast_name_dataset_t& dataset) {
      return in > dataset.names_total_index > dataset.name_to_id > dataset.id_to_name;
   }
   inline void load_fast_name_dataset(const std::string& input) {
      std::stringstream in(input);
      in > fast_name_dataset;
   }
   inline std::string dump_fast_name_dataset() {
      std::stringstream out;
      out < fast_name_dataset;
      return out.str();
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
