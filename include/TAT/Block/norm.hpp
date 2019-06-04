/** TAT/Block/norm.hpp
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

#ifndef TAT_Block_Norm_HPP_
#define TAT_Block_Norm_HPP_

#include "../Block.hpp"

namespace TAT {
  namespace block {
    template<class Base>
    template<int n>
    Block<Base> Block<Base>::norm() const {
      Block<Base> res({});
      res.data = data.template norm<n>();
      return std::move(res);
    } // norm
  } // namespace block
} // namespace TAT

#endif // TAT_Block_Norm_HPP_
