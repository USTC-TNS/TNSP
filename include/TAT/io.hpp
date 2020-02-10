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

namespace TAT {
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
   inline std::ostream& operator<=(std::ostream& out, const Name& name) {
      raw_write(out, &name.id);
      return out;
   }
   inline std::istream& operator>=(std::istream& in, Name& name) {
      raw_read(in, &name.id);
      return in;
   }

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec) {
      out << "[";
      auto not_first = false;
      for (const auto& i : vec) {
         if (not_first) {
            out << ",";
         }
         not_first = true;
         out << i;
      }
      out << "]";
      return out;
   }
   template<class T>
   std::ostream& operator<=(std::ostream& out, const vector<T>& vec) {
      for (const auto& i : vec) {
         out <= i;
      }
      return out;
   }
   template<class T>
   std::istream& operator>=(std::istream& in, vector<T>& vec) {
      for (auto& i : vec) {
         in >= i;
      }
      return in;
   }
   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << edge.map.at(NoSymmetry());
      } else {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out << "{arrow:";
            out << (edge.arrow ? "In" : "Out");
            out << ",map:";
         }
         out << "{";
         auto not_first = false;
         for (const auto& [sym, dim] : edge.map) {
            if (not_first) {
               out << ",";
            }
            not_first = true;
            out << sym << ":" << dim;
         }
         out << "}";
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            out << "}";
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
            out <= sym;
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
            in >= sym;
            raw_read(in, &dim);
            edge.map[sym] = dim;
         }
      }
      return in;
   }

   std::ostream& operator<<(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const NoSymmetry&) {
      return out;
   }
   std::istream& operator>=(std::istream& in, NoSymmetry&) {
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s) {
      out << s.z2;
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const Z2Symmetry& s) {
      raw_write(out, &s.z2);
      return out;
   }
   std::istream& operator>=(std::istream& in, Z2Symmetry& s) {
      raw_read(in, &s.z2);
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const U1Symmetry& s) {
      out << s.u1;
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const U1Symmetry& s) {
      raw_write(out, &s.u1);
      return out;
   }
   std::istream& operator>=(std::istream& in, U1Symmetry& s) {
      raw_read(in, &s.u1);
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const FermiSymmetry& s) {
      out << s.fermi;
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const FermiSymmetry& s) {
      raw_write(out, &s.fermi);
      return out;
   }
   std::istream& operator>=(std::istream& in, FermiSymmetry& s) {
      raw_read(in, &s.fermi);
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& s) {
      out << "(" << s.fermi << "," << s.z2 << ")";
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const FermiZ2Symmetry& s) {
      raw_write(out, &s.fermi);
      raw_write(out, &s.z2);
      return out;
   }
   std::istream& operator>=(std::istream& in, FermiZ2Symmetry& s) {
      raw_read(in, &s.fermi);
      raw_read(in, &s.z2);
      return in;
   }
   std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& s) {
      out << "(" << s.fermi << "," << s.u1 << ")";
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const FermiU1Symmetry& s) {
      raw_write(out, &s.fermi);
      raw_write(out, &s.u1);
      return out;
   }
   std::istream& operator>=(std::istream& in, FermiU1Symmetry& s) {
      raw_read(in, &s.fermi);
      raw_read(in, &s.u1);
      return in;
   }

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry>& tensor) {
      out << "{names:";
      out << tensor.names;
      out << ",edges:";
      out << tensor.core->edges;
      out << ",blocks:";
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         out << tensor.core->blocks.begin()->second;
      } else {
         out << "{";
         auto not_first = false;
         for (const auto& [i, j] : tensor.core->blocks) {
            if (not_first) {
               out << ",";
            }
            not_first = true;
            out << i << ":" << j;
         }
         out << "}";
      }
      out << "}";
      return out;
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>&
   Tensor<ScalarType, Symmetry>::meta_put(std::ostream& out) const {
      auto rank = Rank(names.size());
      raw_write(out, &rank);
      out <= names;
      out <= core->edges;
      return *this;
   }

   template<class ScalarType, class Symmetry>
   const Tensor<ScalarType, Symmetry>&
   Tensor<ScalarType, Symmetry>::data_put(std::ostream& out) const {
      for (const auto& [_, i] : core->blocks) {
         raw_write(out, i.data(), i.size());
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
      Rank rank;
      raw_read(in, &rank);
      names.resize(rank);
      in >= names;
      name_to_index = construct_name_to_index(names);
      vector<Edge<Symmetry>> edges(rank);
      in >= edges;
      core = std::make_shared<Core<ScalarType, Symmetry>>(std::move(edges));
      if (!is_valid_name(names, core->edges.size())) {
         TAT_WARNING("Invalid Names");
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry>& Tensor<ScalarType, Symmetry>::data_get(std::istream& in) {
      for (auto& [_, i] : core->blocks) {
         raw_read(in, i.data(), i.size());
         // TODO: 不同版本之间不兼容, 依赖于map内的less确定的顺序
      }
      return *this;
   }

   template<class ScalarType, class Symmetry>
   std::istream& operator>=(std::istream& in, Tensor<ScalarType, Symmetry>& tensor) {
      tensor.meta_get(in).data_get(in);
      return in;
   }
} // namespace TAT
#endif
