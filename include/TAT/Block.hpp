/** TAT/Block.hpp
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

#ifndef TAT_Block_HPP_
#define TAT_Block_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace block {
    template<Device device, class Base>
    class Block {
     public:
      Block() : dims({}), data() {}

      std::vector<Size> dims;
      Data<device, Base> data;

      ~Block() = default;
      Block(Block<device, Base>&& other) = default;
      Block(const Block<device, Base>& other) = default;
      Block<device, Base>& operator=(Block<device, Base>&& other) = default;
      Block<device, Base>& operator=(const Block<device, Base>& other) = default;
      static Size get_size(const std::vector<Size>& _dims) {
        Size res = 1;
        for (const auto& i : _dims) {
          res *= i;
        } // for i
        return res;
      } // get_size
      template<class T=std::vector<Size>>
      Block(T&& _dims) : data(get_size(_dims)) {
        dims = std::forward<T>(_dims);
      }
      Block(const Base& num) : dims({}), data(num) {}

      const Size& size() const {
        return data.size;
      } // size
      const Base* get() const {
        return data.get();
      } // get
      Base* get() {
        return data.get();
      } // get

      Block<device, Base>& set_test() {
        data.set_test();
        return *this;
      } // set_test
      Block<device, Base>& set_zero() {
        data.set_zero();
        return *this;
      } // set_zero
      Block<device, Base>& set_random(const std::function<Base()>& random) {
        data.set_random(random);
        return *this;
      } // set_random
      Block<device, Base>& set_constant(Base num) {
        data.set_constant(num);
        return *this;
      } // set_constant

      template<class Base2, ENABLE_IF(scalar_tools::is_scalar<Base2>)>
      Block<device, Base2> to() const {
        Block<device, Base2> res;
        res.dims = dims;
        res.data = data.template to<Base2>();
        return std::move(res);
      } // to

      template<int n>
      Block<device, Base> norm() const;

      Block<device, Base> transpose(const std::vector<Rank>& plan) const;

      static Block<device, Base> contract(const Block<device, Base>& block1,
                                          const Block<device, Base>& block2,
                                          const std::vector<Rank>& plan1,
                                          const std::vector<Rank>& plan2,
                                          const Rank& contract_num);

      Block<device, Base> contract(const Block<device, Base>& block2,
                                   const std::vector<Rank>& plan1,
                                   const std::vector<Rank>& plan2,
                                   const Rank& contract_num) const {
        return std::move(Block<device, Base>::contract(*this, block2, plan1, plan2, contract_num));
      } // contract

      Block<device, Base> multiple(const Block<device, Base>& other, const Rank& index) const;

      class svd_res {
       public:
        Block<device, Base> U;
        Block<device, Base> S;
        Block<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const;

      class qr_res {
       public:
        Block<device, Base> Q;
        Block<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const;
    }; // class Block
  } // namespace block
} // namespace TAT

#include "Block/norm.hpp"
#include "Block/transpose.hpp"
#include "Block/contract.hpp"
#include "Block/multiple.hpp"
#include "Block/svd.hpp"
#include "Block/qr.hpp"
#include "Block/scalar.hpp"
#include "Block/io.hpp"

#endif // TAT_Block_HPP_
