/** TAT/Data.hpp
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

#ifndef TAT_Data_HPP_
#define TAT_Data_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
    namespace CPU {
#ifdef TAT_USE_CPU
      template<class Base>
      class Data {
       public:
        Data() : size(0), base() {}

        Size size;
        std::unique_ptr<Base[]> base;

        ~Data() = default;
        Data(Data<Base>&& other) = default;
        Data<Base>& operator=(Data<Base>&& other) = default;
        Data(Size _size) : size(_size) {
          base = std::unique_ptr<Base[]>(new Base[size]);
        }
        Data(const Data<Base>& other) {
          new (this) Data(other.size);
          std::memcpy(get(), other.get(), size*sizeof(Base));
#ifndef TAT_NOT_WARN_COPY
          std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
        }
        Data<Base>& operator=(const Data<Base>& other) {
          new (this) Data(other);
          return *this;
        }
        Data(const Base& num) {
          new (this) Data(Size(1));
          *base.get() = num;
        }

        const Base* get() const {
          return base.get();
        } // get
        Base* get() {
          return base.get();
        } // get

        Data<Base>& set_test() {
          for (Size i=0; i<size; i++) {
            base[i] = Base(i);
          } // for i
          return *this;
        } // set_test
        Data<Base>& set_zero() {
          for (Size i=0; i<size; i++) {
            base[i] = Base(0);
          } // for i
          return *this;
        } // set_zero
        Data<Base>& set_random(const std::function<Base()>& random) {
          for (Size i=0; i<size; i++) {
            base[i] = random();
          } // for i
          return *this;
        } // set_random
        Data<Base>& set_constant(Base num) {
          for (Size i=0; i<size; i++) {
            base[i] = num;
          } // for i
          return *this;
        } // set_constant

        template<class Base2, ENABLE_IF(is_scalar<Base2>)>
        Data<Base2> to() const {
          Data<Base2> res(size);
          for (Size i=0; i<size; i++) {
            res.base[i] = Base2(base[i]);
          } // for i
          return std::move(res);
        } // to

        template<int n>
        Data<Base> norm() const;

        Data<Base> transpose(const std::vector<Size>& dims,
                             const std::vector<Rank>& plan) const;

        static Data<Base> contract(const Data<Base>& data1,
                                   const Data<Base>& data2,
                                   const std::vector<Size>& dims1,
                                   const std::vector<Size>& dims2,
                                   const std::vector<Rank>& plan1,
                                   const std::vector<Rank>& plan2,
                                   const Size& m, const Size& k, const Size& n);

        Data<Base> contract(const Data<Base>& data2,
                            const std::vector<Size>& dims1,
                            const std::vector<Size>& dims2,
                            const std::vector<Rank>& plan1,
                            const std::vector<Rank>& plan2,
                            const Size& m, const Size& k, const Size& n) const {
          return std::move(Data<Base>::contract(*this, data2, dims1, dims2, plan1, plan2, m, k, n));
        } // contract

        Data<Base> multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const;

        class svd_res {
         public:
          Data<Base> U;
          Data<Base> S;
          Data<Base> V;
        }; // class svd_res

        svd_res svd(const std::vector<Size>& dims,
                    const std::vector<Rank>& plan,
                    const Size& u_size,
                    const Size& v_size,
                    const Size& min_mn,
                    const Size& cut) const;

        class qr_res {
         public:
          Data<Base> Q;
          Data<Base> R;
        }; // class qr_res

        qr_res qr(const std::vector<Size>& dims,
                  const std::vector<Rank>& plan,
                  const Size& q_size,
                  const Size& r_size,
                  const Size& min_mn) const;
      }; // class Data
#endif // TAT_USE_CPU
    } // namespace CPU
  } // namespace data
} // namespace TAT

#include "Data/norm.hpp"
#include "Data/transpose.hpp"
#include "Data/contract.hpp"
#include "Data/multiple.hpp"
#include "Data/svd.hpp"
#include "Data/qr.hpp"
#include "Data/scalar.hpp"
#include "Data/io.hpp"

#endif // TAT_Data_HPP_
