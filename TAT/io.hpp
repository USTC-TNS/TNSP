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
#ifndef TAT_IO_HPP_
#   define TAT_IO_HPP_

namespace TAT {
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
      auto pos = id_to_name.find(name.id);
      if (pos == id_to_name.end()) {
         return out << "UserDefinedName" << name.id;
      } else {
         return out << id_to_name.at(name.id);
      }
      return out;
   }
   std::ostream& operator<=(std::ostream& out, const Name& name) {
      raw_write(out, &name.id);
      return out;
   }
   std::istream& operator>=(std::istream& in, Name& name) {
      raw_read(in, &name.id);
      return in;
   }

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec) {
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

   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge) {
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
      return out;
   }
   template<class Symmetry>
   std::ostream& operator<=(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         raw_write(out, &edge.at(NoSymmetry()));
      } else {
         auto nums = Nums(edge.size());
         raw_write(out, &nums);
         for (const auto& [sym, dim] : edge) {
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
         edge[NoSymmetry()] = dim;
      } else {
         Nums nums;
         raw_read(in, &nums);
         edge.clear();
         for (Nums i = 0; i < nums; i++) {
            Symmetry sym;
            Size dim;
            in >= sym;
            raw_read(in, &dim);
            edge[sym] = dim;
         }
      }
      return in;
   }

   template<class ScalarType, class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Block<ScalarType, Symmetry>& block) {
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
      for (const auto& i : core->blocks) {
         raw_write(out, i.raw_data.data(), i.size);
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
      for (auto& i : core->blocks) {
         raw_read(in, i.raw_data.data(), i.size);
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
