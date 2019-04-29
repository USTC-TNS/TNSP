/* TAT/Data/CPU/io.hpp
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

#ifndef TAT_Data_CPU_Io_HPP_
#define TAT_Data_CPU_Io_HPP_

#include "../../Data.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
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
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_CPU_Io_HPP_
