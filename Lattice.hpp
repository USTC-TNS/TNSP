
/* TAT/Lattice.hpp
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

#ifndef TAT_Lattice_HPP_
#define TAT_Lattice_HPP_

#include "TAT.hpp"

namespace TAT {
  namespace lattice {
    template<>
    class Dimension<1> {
     public:
      int i;
    }; // Dimension<1>

    template<>
    class Dimension<2> {
     public:
      int i;
      int j;
    }; // Dimension<2>

    template<>
    class Dimension<3> {
     public:
      int i;
      int j;
      int k;
    }; // Dimension<3>

    template<class Tags, Device device, class Base>
    class Lattice {
      using TensorObj = Tensor<device, Base>;
      using TensorPtr = std::shared_ptr<TensorObj>;
      friend class Link;
      class Link {
       public:
        Tags tag;
        Legs leg;
        TensorPtr env;
      }; // class Link
      std::map<Tags, TensorPtr> site;
      std::map<Tags, std::map<Legs, Link>> bond;

      // use lattice(i, j) = make_site({...}, {...})
      TensorObj& operator[](const Tags& tag) {
        return *site[tag];
      } // operator[]
      template<class ... Args>
      TensorObj& operator()(const Args& ... args) {
        return *site[ {args ...}];
      } // operator()

      // set_bond(.., .., .., ..)
      // set_bond(.., .., .., dim)
      // set_bond(.., .., .., env)
      void set_bond(const Tags& tag1, const Legs& leg1,
                    const Tags& tag2, const Legs& leg2,
                    std::shared_ptr<Tensor<device, Base>> env) {
        bond[tag1][leg1] = {tag2, leg2, env};
        bond[tag2][leg2] = {tag1, leg1, env};
      } // set_bond
      void set_bond(const Tags& tag1, const Legs& leg1,
                    const Tags& tag2, const Legs& leg2,
                    const Size& dims=-1) {
        set_bond(tag1, leg1, tag2, leg2, make_env(dims));
      } // set_bond

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      static TensorPtr make_site(T1&& _legs, T2&& _dims) {
        return std::make_shared<TensorObj>(std::forward<T1>(_legs), std::forward<T2>(_dims));
      } // make_site

      static TensorPtr make_env(const Size& dims=-1) {
        if (dims==-1) {
          return TensorPtr();
        } // if no env
        auto res = std::make_shared<TensorObj>({legs_name::Phy}, {dims});
        res->set_constant(1);
        return res;
      } // make_env

      // void operator<<
      // void operator<<
      // void operator>>

      // inplace
      // set_...
      void norm();
      void norm_env();

      // no inplace, maybe need copy first
      // env
      void emit_env();
      void absorb_env();

      // low level
      void contract();
      void svd();
      void qr();

      // high level
      void qr_svd();
      void update();
      void qr_update();
      void qr_to();
      void eat();
    }; // class Lattice
  } // namespace lattice
} // namespace TAT

#endif // TAT_Lattice_HPP
