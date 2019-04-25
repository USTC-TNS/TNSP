/* TAT/Node/io.hpp
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

#ifndef TAT_Node_Io_HPP_
#define TAT_Node_Io_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    inline namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Size>& value) {
        Rank size=value.size();
        out << "[";
        for (Rank i=0; i<size; i++) {
          out << value[i];
          if (i!=size-1) {
            out << ", ";
          } // if not last
        } // for i
        out << "]";
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Node<device, Base>& value) {
        return out << "{" << rang::fg::magenta << "\"dims\": " << value.dims << rang::fg::reset << ", \"data\": " << value.data << "}";
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Node<device, Base>& value) {
        Rank rank = value.dims.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.dims.data(), rank*sizeof(Size));
        out << value.data;
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Node<device, Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.dims.resize(rank);
        in.read((char*)value.dims.data(), rank*sizeof(Size));
        in >> value.data;
        return in;
      } // operator<<
    } // namespace node::io
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Io_HPP_
