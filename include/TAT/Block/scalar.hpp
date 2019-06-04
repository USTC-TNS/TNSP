/** TAT/Block/scalar.hpp
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

#ifndef TAT_Block_Scalar_HPP_
#define TAT_Block_Scalar_HPP_

#include "../Block.hpp"

namespace TAT {
  namespace block {
    inline namespace scalar {
      bool operator==(const std::vector<Size>& a, const std::vector<Size>& b) {
        if (a.size()!=b.size()) {
          return false;
        } // if size
        Rank size=a.size();
        for (Rank i=0; i<size; i++) {
          if (a[i]!=b[i]) {
            return false;
          } // if
        } // for i
        return true;
      } // operator==

      template<class Base>
      Block<Base>& operator*=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data *= b.data;
        return a;
      } // operator*=

      template<class Base>
      Block<Base> operator*(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data * b.data;
        return std::move(res);
      } // operator*

      template<class Base>
      Block<Base>& operator/=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data /= b.data;
        return a;
      } // operator/=

      template<class Base>
      Block<Base> operator/(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data / b.data;
        return std::move(res);
      } // operator/

      template<class Base>
      Block<Base> operator+(const Block<Base>& a) {
        Block<Base> res;
        res.dims = a.dims;
        res.data = + a.data;
        return std::move(res);
      } // operator+

      template<class Base>
      Block<Base> operator+(Block<Base>&& a) {
        Block<Base> res;
        res.dims = std::move(a.dims);
        res.data = + std::move(a.data);
        return std::move(res);
      } // operator+

      template<class Base>
      Block<Base>& operator+=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data += b.data;
        return a;
      } // operator+=

      template<class Base>
      Block<Base> operator+(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data + b.data;
        return std::move(res);
      } // operator+

      template<class Base>
      Block<Base> operator-(const Block<Base>& a) {
        Block<Base> res;
        res.dims = a.dims;
        res.data = - a.data;
        return std::move(res);
      } // operator-

      template<class Base>
      Block<Base>& operator-=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data -= b.data;
        return a;
      } // operator-=

      template<class Base>
      Block<Base> operator-(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data - b.data;
        return std::move(res);
      } // operator-
    } // namespace block::scalar
  } // namespace block
} // namespace TAT

#endif // TAT_Block_Scalar_HPP_
