/** TAT/transpose.hpp
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

#ifndef TAT_Transpose_HPP_
#define TAT_Transpose_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      namespace transpose {
        template<class Base>
        void run(const std::vector<Rank>& plan, const std::vector<Size>& dims, const Base* src, Base* dst) {
          std::vector<int> int_plan(plan.begin(), plan.end());
          std::vector<int> int_dims(dims.begin(), dims.end());
          hptt::create_plan(int_plan.data(), int_plan.size(),
                            1, src, int_dims.data(), NULL,
                            0, dst, NULL,
                            hptt::ESTIMATE, 1, NULL, 1)->execute();
        } // run
      } // namespace data::CPU::transpose

      template<class Base>
      Data<Base> Data<Base>::transpose(const std::vector<Size>& dims,
                                       const std::vector<Rank>& plan) const {
        Data<Base> res(size);
        assert(dims.size()==plan.size());
        transpose::run(plan, dims, get(), res.get());
        return std::move(res);
      } // transpose
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

namespace TAT {
  namespace block {
    namespace transpose {
      void plan(std::vector<Size>& new_dims, const std::vector<Size>& dims, const std::vector<Rank>& plan) {
        for (const auto& i : plan) {
          new_dims.push_back(dims[i]);
        } // for i
      } // plan
    } // namespace block::transpose

    template<class Base>
    Block<Base> Block<Base>::transpose(const std::vector<Rank>& plan) const {
      Block<Base> res;
      transpose::plan(res.dims, dims, plan);
      assert(plan.size()==dims.size());
      assert(get_size(res.dims)==data.size);
      res.data = data.transpose(dims, plan);
      return std::move(res);
    } // transpose
  } // namespace block
} // namespace TAT

namespace TAT {
  namespace node {
    namespace transpose {
      void plan(std::vector<Rank>& plan, const std::vector<Legs>& new_legs, const std::vector<Legs>& legs) {
        const Rank& rank = legs.size();
        for (Rank i=0; i<rank; i++) {
          for (Rank j=0; j<rank; j++) {
            if (new_legs[i]==legs[j]) {
              plan.push_back(j);
              break;
            } // if
          } // for j
        } // for i
      } // plan
    } // namespace node::transpose

    template<class Base>
    Node<Base> Node<Base>::transpose(const std::vector<Legs>& new_legs) const {
      Node<Base> res;
      res.legs = internal::in_and_in(new_legs, legs);
      assert(legs.size()==res.legs.size());
#ifndef NDEBUG
      auto set_new = std::set<Legs>(res.legs.begin(), res.legs.end());
      assert(set_new.size()==res.legs.size());
      set_new.insert(legs.begin(), legs.end());
      assert(set_new.size()==res.legs.size());
#endif // NDEBUG
      std::vector<Rank> plan;
      transpose::plan(plan, res.legs, legs);
      assert(res.legs.size()==legs.size());
      assert(plan.size()==legs.size());
      res.tensor = tensor.transpose(plan);
      return std::move(res);
    } // transpose
  } // namespace node
} // namespace TAT

#endif // TAT_Transpose_HPP_
