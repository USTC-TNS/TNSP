/** TAT/io.hpp
 * @file
 * @author  Hao Zhang <zh970204@mail.ustc.edu.cn>
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

#ifndef TAT_Io_HPP_
#define TAT_Io_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      inline namespace io {
        template<class Base>
        std::ostream& operator<<(std::ostream& out, const Data<Base>& value) {
          out << "{\"" << rang::fgB::green << "size\": " << value.size << "" << rang::fg::reset << ", " << rang::fg::yellow << "\"base\": [";
          if (value.size!=0) {
            for (Size i=0; i<value.size-1; i++) {
              out << value.base[i] << ", ";
            } // for i
            out << value.base[value.size-1];
          } // if
          out << "]" << rang::fg::reset << "}";
          return out;
        } // operator<<

        template<class Base>
        std::ofstream& operator<<(std::ofstream& out, const Data<Base>& value) {
          out.write((char*)&value.size, sizeof(Size));
          out.write((char*)value.get(), value.size*sizeof(Base));
          return out;
        } // operator<<

        template<class Base>
        std::ifstream& operator>>(std::ifstream& in, Data<Base>& value) {
          in.read((char*)&value.size, sizeof(Size));
          value.base = std::unique_ptr<Base[]>(new Base[value.size]);
          in.read((char*)value.get(), value.size*sizeof(Base));
          return in;
        } // operator<<
      } // namespace data::CPU::io
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

namespace TAT {
  namespace block {
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

      template<class Base>
      std::ostream& operator<<(std::ostream& out, const Block<Base>& value) {
        return out << "{" << rang::fg::magenta << "\"dims\": " << value.dims << rang::fg::reset << ", \"data\": " << value.data << "}";
      } // operator<<

      template<class Base>
      std::ofstream& operator<<(std::ofstream& out, const Block<Base>& value) {
        Rank rank = value.dims.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.dims.data(), rank*sizeof(Size));
        out << value.data;
        return out;
      } // operator<<

      template<class Base>
      std::ifstream& operator>>(std::ifstream& in, Block<Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.dims.resize(rank);
        in.read((char*)value.dims.data(), rank*sizeof(Size));
        in >> value.data;
        return in;
      } // operator<<
    } // namespace block::io
  } // namespace block
} // namespace TAT

namespace TAT {
  namespace node {
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

      template<class Base>
      std::ostream& operator<<(std::ostream& out, const Node<Base>& value) {
        return out << "{" << rang::fgB::yellow << "\"rank\": " << value.legs.size() << rang::fg::reset << ", " << rang::fgB::blue << "\"legs\": " << value.legs << rang::fg::reset << ", \"tensor\": " << value.tensor << "}";
      } // operator<<

      template<class Base>
      std::ofstream& operator<<(std::ofstream& out, const Node<Base>& value) {
        Rank rank = value.legs.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.legs.data(), rank*sizeof(Legs));
        out << value.tensor;
        return out;
      } // operator<<

      template<class Base>
      std::ifstream& operator>>(std::ifstream& in, Node<Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.legs.resize(rank);
        in.read((char*)value.legs.data(), rank*sizeof(Legs));
        in >> value.tensor;
        return in;
      } // operator<<
    } // namespace node::io
  } // namespace node
} // namespace TAT

#endif // TAT_Io_HPP_
