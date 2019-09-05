/**
 * \file lazy_TAT.hpp
 *
 * Copyright (C) 2019  Hao Zhang <zh970205@mail.ustc.edu.cn>
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

#ifndef TAT_LAZY_TAT_HPP_
#define TAT_LAZY_TAT_HPP_

#include <TAT.hpp>
#include <lazy.hpp>

namespace TAT {
      /**
       * lazy化的Node
       */
      namespace lazy_node {
            template<class Base>
            lazy::maybe_ptr<const TAT::Node<Base>>
            transpose_nocpy(const TAT::Node<Base>& self, const std::vector<Legs>& new_legs) {
                  if (TAT::node::operator==(self.legs, new_legs)) {
                        return {&self, false};
                  } else {
                        auto res = self.transpose(new_legs);
                        return {new TAT::Node<Base>(std::move(res)), true};
                  }
            }

            template<class Base, class Base2>
            lazy::maybe_ptr<const TAT::Node<Base>>
            multiple_nocpy(const TAT::Node<Base>& self, const TAT::Node<Base2>& other, const Legs& position) {
                  assert(other.legs.size() == 1);
                  auto pos = std::find(self.legs.begin(), self.legs.end(), position);
                  if (pos == self.legs.end()) {
                        return {&self, false};
                  } else {
                        auto res = self.multiple(other, position);
                        return {new TAT::Node<Base>(std::move(res)), true};
                  }
            }

            template<class, int>
            struct LazyNode;

            template<class T>
            struct find_base;
            template<class Base>
            struct find_base<TAT::Node<Base>> {
                  using type = Base;
            };
            template<class T>
            struct find_base<lazy::Lazy<T>> {
                  using type = typename find_base<T>::type;
            };
            template<class T, int N>
            struct find_base<LazyNode<T, N>> {
                  using type = T;
            };
            template<class T>
            using find_base_t = typename find_base<T>::type;

            /**
             * Node的lazy化, 只需要对每个操作各自写一边封装即可
             *
             * \see TAT::node::Node
             * \see lazy::Lazy
             */
            template<class Base = double, int N = 1>
            struct LazyNode : lazy::HighLazy<TAT::Node<Base>, N> {
                  using type = Base;
                  using lazy_node_base = lazy::HighLazy<TAT::Node<Base>, N>;
                  using element_type = typename lazy_node_base::element_type;
                  using self = LazyNode<Base, N>;

                  using lazy_node_base::lazy_node_base;
                  /**
                   * 因Node的构造中使用了类型不识别的vector<Size>, vector<Legs>, 所以需要单独写出来
                   */
                  template<
                        class Legs = std::vector<TAT::Legs>,
                        class Dims = std::vector<TAT::Size>,
                        class = std::enable_if_t<
                              std::is_same_v<std::remove_cv_t<std::remove_reference_t<Legs>>, std::vector<TAT::Legs>> &&
                              std::is_same_v<std::remove_cv_t<std::remove_reference_t<Dims>>, std::vector<TAT::Size>>>>
                  LazyNode(Legs&& legs, Dims&& dims) :
                        lazy_node_base(std::forward<Legs>(legs), std::forward<Dims>(dims)) {}

                  self legs_rename(const std::map<Legs, Legs>& dict) const {
                        return lazy::Lazy(&element_type::legs_rename, lazy::exec(*this), dict);
                  }

                  template<class Base2>
                  self to() const {
                        return lazy::Lazy(&element_type::template to<Base2>, lazy::exec(*this));
                  }

                  template<int n>
                  self norm() const {
                        return lazy::Lazy(&element_type::template norm<n>, lazy::exec(*this));
                  }

                  self transpose(const std::vector<Legs>& new_legs) const {
                        if constexpr (N == 1) {
                              return lazy::Lazy(&transpose_nocpy<type>, lazy::exec(*this), new_legs);
                        } else {
                              return lazy::Lazy(&element_type::transpose, lazy::exec(*this), new_legs);
                        }
                  }

                  struct svd_res {
                        LazyNode<Base, N> U;
                        LazyNode<real_base_t<Base>, N> S;
                        LazyNode<Base, N> V;
                  };

                  svd_res
                  svd(const std::vector<Legs>& input_u_legs,
                      const Legs& new_u_legs,
                      const Legs& new_v_legs,
                      const Rank& cut = -1) const {
                        auto tmp = lazy::Lazy(
                              &element_type::svd, lazy::exec(*this), input_u_legs, new_u_legs, new_v_legs, cut);
                        auto res = svd_res();
                        res.U = lazy::Lazy(&element_type::svd_res::u, lazy::exec(tmp));
                        res.S = lazy::Lazy(&element_type::svd_res::s, lazy::exec(tmp));
                        res.V = lazy::Lazy(&element_type::svd_res::v, lazy::exec(tmp));
                        return res;
                  }

                  struct qr_res {
                        LazyNode<Base, N> Q;
                        LazyNode<Base, N> R;
                  };

                  qr_res
                  qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
                        auto tmp =
                              lazy::Lazy(&element_type::qr, lazy::exec(*this), input_q_legs, new_q_legs, new_r_legs);
                        auto res = qr_res();
                        res.Q = lazy::Lazy(&element_type::qr_res::q, lazy::exec(tmp));
                        res.R = lazy::Lazy(&element_type::qr_res::r, lazy::exec(tmp));
                        return res;
                  }

                  qr_res
                  rq(const std::vector<Legs>& input_r_legs, const Legs& new_r_legs, const Legs& new_q_legs) const {
                        auto tmp =
                              lazy::Lazy(&element_type::rq, lazy::exec(*this), input_r_legs, new_r_legs, new_q_legs);
                        auto res = qr_res();
                        res.Q = lazy::Lazy(&element_type::qr_res::q, lazy::exec(tmp));
                        res.R = lazy::Lazy(&element_type::qr_res::r, lazy::exec(tmp));
                        return res;
                  }

                  static self contract(
                        const self& node1,
                        const self& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2) {
                        return lazy::Lazy(&element_type::contract, lazy::exec(node1), lazy::exec(node2), legs1, legs2);
                  }

                  static self contract(
                        const self& node1,
                        const element_type& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2) {
                        return lazy::Lazy(&element_type::contract, lazy::exec(node1), node2, legs1, legs2);
                  }

                  static self contract(
                        const element_type& node1,
                        const self& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2) {
                        return lazy::Lazy(&element_type::contract, node1, lazy::exec(node2), legs1, legs2);
                  }

                  template<class Arg, class = std::enable_if_t<std::is_base_of_v<Arg, LazyNode<find_base_t<Arg>, N>>>>
                  self multiple(const Arg& other, const Legs& position) const {
                        if constexpr (N == 1) {
                              return lazy::Lazy(
                                    &multiple_nocpy<type, typename Arg::element_type::type>,
                                    lazy::exec(*this),
                                    lazy::exec(other),
                                    position);
                        } else {
                              return lazy::Lazy(
                                    &element_type::template multiple<typename Arg::element_type>,
                                    lazy::exec(*this),
                                    lazy::exec(other),
                                    position);
                        }
                  }
            };

            template<class Base, int N>
            struct LazyNodeHelper {
                  using type = LazyNode<Base, N>;
            };
            template<class Base>
            struct LazyNodeHelper<Base, 0> {
                  using type = lazy::HighLazy<TAT::Node<Base>, 0>;
            };
      } // namespace lazy_node
      template<class Base = double, int N = 1>
      using LazyNode = typename lazy_node::LazyNodeHelper<Base, N>::type;
} // namespace TAT

#endif // TAT_LAZY_TAT_HPP_