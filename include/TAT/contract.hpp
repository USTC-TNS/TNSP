/** TAT/contract.hpp
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

#ifndef TAT_Contract_HPP_
#define TAT_Contract_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      namespace contract {
        template<class Base>
        void run(Base* data,
                 const Base* data1,
                 const Base* data2,
                 const Size& m,
                 const Size& n,
                 const Size& k);

        template<>
        void run<float>(float* data,
                        const float* data1,
                        const float* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      1, const_cast<float*>(data1), k, const_cast<float*>(data2), n,
                      0, data, n);
        } // run<float>

        template<>
        void run<double>(double* data,
                         const double* data1,
                         const double* data2,
                         const Size& m,
                         const Size& n,
                         const Size& k) {
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      1, const_cast<double*>(data1), k, const_cast<double*>(data2), n,
                      0, data, n);
        } // run<double>

        template<>
        void run<std::complex<float>>(std::complex<float>* data,
                                      const std::complex<float>* data1,
                                      const std::complex<float>* data2,
                                      const Size& m,
                                      const Size& n,
        const Size& k) {
          std::complex<float> alpha = 1;
          std::complex<float> beta = 0;
          cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      &alpha, const_cast<std::complex<float>*>(data1), k, const_cast<std::complex<float>*>(data2), n,
                      &beta, data, n);
        } // run<std::complex<float>>

        template<>
        void run<std::complex<double>>(std::complex<double>* data,
                                       const std::complex<double>* data1,
                                       const std::complex<double>* data2,
                                       const Size& m,
                                       const Size& n,
        const Size& k) {
          std::complex<double> alpha = 1;
          std::complex<double> beta = 0;
          cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      &alpha, const_cast<std::complex<double>*>(data1), k, const_cast<std::complex<double>*>(data2), n,
                      &beta, data, n);
        } // run<std::complex<double>>
      } // namespace data::CPU::contract

      template<class Base>
      Data<Base> Data<Base>::contract(const Data<Base>& data1,
                                      const Data<Base>& data2,
                                      const std::vector<Size>& dims1,
                                      const std::vector<Size>& dims2,
                                      const std::vector<Rank>& plan1,
                                      const std::vector<Rank>& plan2,
                                      const Size& m, const Size& k, const Size& n) {
        assert(m*k==data1.size);
        assert(k*n==data2.size);
        Data<Base> a = data1.transpose(dims1, plan1);
        Data<Base> b = data2.transpose(dims2, plan2);
        // wasted transpose
        Data<Base> res(m*n);
        contract::run<Base>(res.get(), a.get(), b.get(), m, n, k);
        return std::move(res);
      } // contract
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

namespace TAT {
  namespace block {
    namespace contract {
      void plan(std::vector<Size>& dims, Size& m, Size& k, Size& n,
                const::std::vector<Size>& dims1,
                const::std::vector<Size>& dims2,
                const std::vector<Rank>& plan1,
                const std::vector<Rank>& plan2,
                const Rank& contract_num) {
        Rank i, tmp=dims1.size()-contract_num, rank2=dims2.size();
        for (i=0; i<tmp; i++) {
          const Size& t = dims1[plan1[i]];
          m *= t;
          dims.push_back(t);
        } // for i
        for (i=0; i<contract_num; i++) {
          k *= dims1[plan1[i+tmp]];
          assert(dims1[plan1[i+tmp]]==dims2[plan2[i]]);
        } // for i
        for (; i<rank2; i++) {
          const Size& t = dims2[plan2[i]];
          n *= t;
          dims.push_back(t);
        } // for i
      } // plan
    } // namespace block::contract

    template<class Base>
    Block<Base> Block<Base>::contract(const Block<Base>& block1,
        const Block<Base>& block2,
        const std::vector<Rank>& plan1,
        const std::vector<Rank>& plan2,
        const Rank& contract_num) {
      Block<Base> res;
      Size m=1, k=1, n=1;
      contract::plan(res.dims, m, k, n, block1.dims, block2.dims, plan1, plan2, contract_num);
      res.data = Data<Base>::contract(block1.data, block2.data, block1.dims, block2.dims, plan1, plan2, m, k, n);
      return std::move(res);
    } // contract
  } // namespace block
} // namespace TAT

namespace TAT {
  namespace node {
    namespace contract {
      void plan(Rank& contract_num,
                std::vector<Legs>& legs,
                std::vector<Legs>& new_legs1,
                std::vector<Legs>& new_legs2,
                const std::vector<Legs>& total_legs1,
                const std::vector<Legs>& total_legs2,
                const std::vector<Legs>& legs1,
                const std::vector<Legs>& legs2,
                const std::map<Legs, Legs>& map1,
                const std::map<Legs, Legs>& map2) {
        auto filt_legs1 = internal::in_and_not_in(total_legs1, legs1);
        internal::append(new_legs1, filt_legs1);
        internal::append(legs, internal::map_or_not(filt_legs1, map1));

        auto tmp_legs1 = internal::in_and_in(legs1, total_legs1);
        internal::append(new_legs1, tmp_legs1);

        auto tmp_legs2 = internal::in_and_in(legs2, total_legs2);
        internal::append(new_legs2, tmp_legs2);

        auto filt_legs2 = internal::in_and_not_in(total_legs2, legs2);
        internal::append(new_legs2, filt_legs2);
        internal::append(legs, internal::map_or_not(filt_legs2, map2));

        assert(tmp_legs1.size()==tmp_legs2.size());
        contract_num = tmp_legs1.size();
      } // plan
    } // namespace node::contract

    template<class Base>
    Node<Base> Node<Base>::contract(const Node<Base>& node1,
        const Node<Base>& node2,
        const std::vector<Legs>& legs1,
        const std::vector<Legs>& legs2,
        const std::map<Legs, Legs>& map1,
        const std::map<Legs, Legs>& map2) {
      Node<Base> res;
      std::vector<Legs> new_legs1, new_legs2;
      std::vector<Rank> plan1, plan2;
      Rank contract_num;
      assert(legs1.size()==legs2.size());
      contract::plan(contract_num, res.legs, new_legs1, new_legs2, node1.legs, node2.legs, legs1, legs2, map1, map2);
      transpose::plan(plan1, new_legs1, node1.legs);
      transpose::plan(plan2, new_legs2, node2.legs);
      assert(new_legs1.size()==node1.legs.size());
      assert(plan1.size()==node1.legs.size());
      assert(new_legs2.size()==node2.legs.size());
      assert(plan2.size()==node2.legs.size());
      res.tensor = Tensor<Base>::contract(node1.tensor, node2.tensor, plan1, plan2, contract_num);
      return std::move(res);
    } // contract
  } // namespace node
} // namespace TAT

#endif // TAT_Contract_HPP_
