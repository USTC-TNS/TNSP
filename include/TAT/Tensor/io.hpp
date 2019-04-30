/* TAT/Tensor/io.hpp
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

#ifndef TAT_Tensor_Io_HPP_
#define TAT_Tensor_Io_HPP_

#include "../Tensor.hpp"

namespace TAT {
  namespace tensor {
    inline namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Legs>& value) {
        Rank size=value.size();
        out << "[";
        for (Rank i=0; i<size; i++) {
          out << "\"" << value[i] << "\"";
          if (i!=size-1) {
            out << ", ";
          } // if not last
        } // for i
        out << "]";
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Tensor<device, Base>& value) {
        return out << "{" << rang::fgB::yellow << "\"rank\": " << value.legs.size() << rang::fg::reset << ", " << rang::fgB::blue << "\"legs\": " << value.legs << rang::fg::reset << ", \"node\": " << value.node << "}";
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Tensor<device, Base>& value) {
        Rank rank = value.legs.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.legs.data(), rank*sizeof(Legs));
        out << value.node;
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Tensor<device, Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.legs.resize(rank);
        in.read((char*)value.legs.data(), rank*sizeof(Legs));
        in >> value.node;
        return in;
      } // operator<<
    } // namespace tensor::io
  } // namespace tensor
} // namespace TAT

#endif // TAT_Tensor_Io_HPP_
