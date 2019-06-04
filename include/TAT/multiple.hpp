/** TAT/multiple.hpp
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

#ifndef TAT_Multiple_HPP_
#define TAT_Multiple_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      namespace multiple {
        template<class Base>
        void run(Base* res_data, const Base* src_data, const Base* other_data, const Size& a, const Size& b, const Size& c) {
          for (Size i=0; i<a; i++) {
            for (Size j=0; j<b; j++) {
              Base v = other_data[j];
              for (Size k=0; k<c; k++) {
                *(res_data++) = *(src_data++) * v;
              } // for k
            } // for j
          } // for i
        } // run
      } // namespace data::CPU::multiple

      template<class Base>
      Data<Base> Data<Base>::multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const {
        Data<Base> res(size);
        assert(b==other.size);
        assert(a*b*c==size);
        multiple::run<Base>(res.get(), get(), other.get(), a, b, c);
        return std::move(res);
      } // multiple
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

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

namespace TAT {
  namespace node {
    template<class Base>
    Node<Base> Node<Base>::multiple(const Node<Base>& other, const Legs& position) const {
      Node<Base> res;
      assert(other.legs.size()==1);
      res.legs = legs;
      auto pos = std::find(legs.begin(), legs.end(), position);
      if (pos==legs.end()) {
        return *this;
      } // if not multiple
      Rank index = std::distance(legs.begin(), pos);
      res.tensor = tensor.multiple(other.tensor, index);
      return std::move(res);
    } // multiple
  } // namespace node
} // namespace TAT

#endif // TAT_Multiple_HPP_
