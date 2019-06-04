/** TAT/Block/multiple.hpp
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

#ifndef TAT_Block_Multiple_HPP_
#define TAT_Block_Multiple_HPP_

#include "../Block.hpp"

namespace TAT {
  namespace block {
    namespace multiple {
      void plan(Size& a, Size& b, Size& c, const std::vector<Size>& dims, const Rank& index) {
        Rank i=0, rank=dims.size();
        for (; i<index; i++) {
          a *= dims[i];
        } // for i
        b = dims[i];
        i++;
        for (; i<rank; i++) {
          c *= dims[i];
        } // for
      } // plan
    } // namespace block::multiple

    template<class Base>
    Block<Base> Block<Base>::multiple(const Block<Base>& other, const Rank& index) const {
      Block<Base> res;
      res.dims = dims;
      Size a=1, b=1, c=1;
      multiple::plan(a, b, c, dims, index);
      assert(other.dims.size()==1);
      assert(b==other.dims[0]);
      res.data = data.multiple(other.data, a, b, c);
      return std::move(res);
    } // multiple
  } // namespace block
} // namespace TAT

#endif // TAT_Block_Multple_HPP_
