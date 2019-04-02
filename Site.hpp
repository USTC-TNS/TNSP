/* TAT/Site.hpp
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

#ifndef TAT_Site_HPP_
#define TAT_Site_HPP_

#include "TAT.hpp"

namespace TAT {
  namespace site {
    namespace internal {
      template<class T>
      Rank get_index(const std::vector<T>& v, const T& j) {
        auto pos = std::find(v.begin(), v.end(), j);
        assert(pos!=v.end());
        return std::distance(v.begin(), pos);
      } // get_index

      template<Device device, class Base>
      std::shared_ptr<const Tensor<device, Base>> new_env(const Size& dim) {
        auto env = std::shared_ptr<const Tensor<device, Base>>(new Tensor<device, Base>({Legs::Phy}, {dim}));
        const_cast<Tensor<device, Base>&>(*env).set_constant(1);
        return env;
      } // new_env

      template<class T>
      T replace_or_not(const std::map<T, T>& m, const T& k) {
        T res = k;
        try {
          res = m.at(k);
        } catch (const std::out_of_range& e) {
        } // try
        return res;
      } // replace_or_not

      template<class T>
      std::vector<T> vector_except(const std::vector<T>& v, const T& j) {
        std::vector<Legs> res;
        for (const auto& i : v) {
          if (i!=j) {
            res.push_back(i);
          }
        }
        return res;
      } // vector_except

      template<class T>
      void vector_replace_one(std::vector<T>& v, const T& j, const T& k) {
        for (auto& i : v) {
          if (i==j) {
            i = k;
            return;
          }
        }
      } // vector_replace_one

      template<class T>
      bool in_vector(const T& i, const std::vector<T>& v) {
        auto pos = std::find(v.begin(), v.end(), i);
        return pos!=v.end();
      } // in_vector
    } // namespace internal

    // Site won't change Tensor itself, but allow user change it
    // usually, it only replace Tensor rather than change it
    // user should not change Tensor too except initialize or normalize
    // in other word, keep the meaning of Tensor
    template<Device device, class Base>
    class Site {
     public:
      friend class Edge;
      class Edge {
       public:
        const Site<device, Base>* _site;
        Legs legs;
        std::shared_ptr<const Tensor<device, Base>> _env;

        static Edge make_edge(const Site<device, Base>& site_ref, const Legs& _legs) {
          Edge res;
          res.link(site_ref, _legs);
          return res;
        } // make_edge

        Site<device, Base>& site() const {
          return const_cast<Site<device, Base>&>(*_site);
        } // site
        Tensor<device, Base>& env() const {
          return const_cast<Tensor<device, Base>&>(*_env.get());
        } // env

        Edge& link(const Site<device, Base>& site_ref) {
          _site = &site_ref;
          return *this;
        } // link
        Edge& link(const Site<device, Base>& site_ref, const Legs& _legs) {
          _site = &site_ref;
          legs = _legs;
          return *this;
        } // link
        Edge& set(Tensor<device, Base>&& t) {
          _env = std::make_shared<const Tensor<device, Base>>(std::move(t));
          return *this;
        } // set
        Edge& set(std::shared_ptr<const Tensor<device, Base>>& t) {
          _env = t;
          return *this;
        } // set
      }; // class Edge

      std::shared_ptr<const Tensor<device, Base>> _tensor;
      std::map<Legs, Edge> neighbor;

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      static Site<device, Base> make_site(T1&& _legs, T2&& _dims) {
        Site<device, Base> res;
        res.set(Tensor<device, Base>(std::forward<T1>(_legs), std::forward<T2>(_dims)));
        return std::move(res);
      } // make_site

      Tensor<device, Base>& tensor() const {
        return const_cast<Tensor<device, Base>&>(*_tensor.get());
      } // tensor
      const Edge& operator()(const Legs& legs) const {
        return neighbor[legs];
      } // operator()
      Edge& operator()(const Legs& legs) {
        return neighbor[legs];
      } // operator()

      Site<device, Base>& set(Tensor<device, Base>&& t) {
        _tensor = std::make_shared<Tensor<device, Base>>(std::move(t));
        return *this;
      } // set
      Site<device, Base>& set(std::shared_ptr<const Tensor<device, Base>>& t) {
        _tensor = t;
        return *this;
      } // set

      const std::vector<Legs>& legs() const  {
        return tensor().legs;
      } // legs
      const std::vector<Size>& dims() const {
        return tensor().dims();
      } // dims
      const Size& size() const {
        return tensor().size();
      } // size

      // absorb env and emit
      void absorb_env(const Legs& leg, const Edge& edge) {
        if (edge._env) {
          set(std::move(tensor().multiple(edge.env(), leg)));
        } // if env
      } // absorb_env

      void absorb_env(const Legs& leg) {
        auto edge = neighbor[leg];
        absorb_env(leg, edge);
      } // absorb_env

      void emit_env(const Legs& leg, const Edge& edge) {
        if (edge._env) {
          set(std::move(tensor().multiple(1/edge.env(), leg)));
        } // if env
      } // emit_env

      void emit_env(const Legs& leg) {
        auto edge = neighbor[leg];
        emit_env(leg, edge);
      } // emit_env

      void absorb_all() {
        for (const auto& i : neighbor) {
          if (i.second._env) {
            set(std::move(tensor().multiple(i.second.env(), i.first)));
          } // if env
        } // for edge
      } // absorb_all

      void emit_all() {
        for (const auto& i : neighbor) {
          if (i.second._env) {
            set(std::move(tensor().multiple(1/i.second.env(), i.first)));
          } // if env
        } // for edge
      } // emit_all

      // link/unlink * with_env/without_env * single/double
      // single is member function and double is always static function
      std::shared_ptr<const Tensor<device, Base>> create_env_for_leg(const Legs& leg) {
        auto pos = std::find(legs().begin(), legs().end(), leg);
        auto index = std::distance(legs().begin(), pos);
        auto dim = dims()[index];
        auto env = std::shared_ptr<const Tensor<device, Base>>(new Tensor<device, Base>({Legs::Phy}, {dim}));
        const_cast<Tensor<device, Base>&>(*env).set_constant(1);
        return env;
      } // create_env_for_leg

      void link(const Legs& legs1, const Site<device, Base>& site2, const Legs& legs2) {
        neighbor[legs1] = Edge::make_edge(site2, legs2);
      } // link, single link

      static void link(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        site1.link(legs1, site2, legs2);
        site2.link(legs2, site1, legs1);
      } // link, double link, return dim

      void link_env(const Legs& legs1, const Site<device, Base>& site2, const Legs& legs2, std::shared_ptr<const Tensor<device, Base>> env, bool emit=true) {
        link(legs1, site2, legs2);
        neighbor[legs1].set(env);
        if (emit) {
          emit_env(legs1);
        } // if emit
      } // link_env, single link

      static void link_env(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2, std::shared_ptr<const Tensor<device, Base>> env) {
        site1.link_env(legs1, site2, legs2, env, true);
        site2.link_env(legs2, site1, legs1, env, false);
      } // link_env, double link, insert env, change site1

      void unlink(const Legs& legs1) {
        neighbor.erase(legs1);
      } // unlink, single unlink

      static void unlink(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        site1.unlink(legs1);
        site2.unlink(legs2);
      } // unlink, double unlink

      void unlink_env(const Legs& legs1, const bool& absorb=true) {
        if (absorb) {
          absorb_env(legs1);
        }
        unlink(legs1);
      } // unlink, single unlink, delete env

      static void unlink_env(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        site1.unlink_env(legs1, true);
        site2.unlink_env(legs2, false);
      } // unlink, double unlink, delete env, change site1

      // io
      // site does not implement file write, write tensor directly since addr of site is not fixed
      friend std::ostream& operator<<(std::ostream& out, const Site<device, Base>& value) {
        out << "{\"addr\": \"" << &value << "\", \"neighbor\": {";
        bool flag=false;
        for (const auto& i : value.neighbor) {
          if (flag) {
            out << ", ";
          } // if flag
          out << "\"" << i.first << "\": " << "{\"addr\": \"" << &i.second.site() << "\", \"legs\": \"" << i.second.legs << "\"";
          if (i.second._env) {
            out << ", \"env\": ";
            out << i.second.env();
          } // if env
          out << "}";
          flag = true;
        } // for
        out << "} ,\"tensor\": " << value.tensor() << "}";
        return out;
      } // operator<<

      // normalize and other inplace op
      Site<device, Base>& set_test() {
        tensor().set_test();
        return *this;
      } // set_test
      Site<device, Base>& set_zero() {
        tensor().set_zero();
        return *this;
      } // set_zero
      Site<device, Base>& set_random(Base(*random)()) {
        tensor().set_random(random);
        return *this;
      } // set_random
      Site<device, Base>& set_constant(Base num) {
        tensor().set_constant(num);
        return *this;
      } // set_constant

      template<int n>
      Site<device, Base>& normalize() {
        tensor() /= tensor().template norm<n>();
        return *this;
      } // normalize

      // middle level op
      // norm add scalar operated onto tensor directly
      // so, we need implement svd, qr and contract only
      // it have nearly the same parameter to tensor
      // but it use subroutine always instead of function
      // since it need to know address of site
      // higher level op see below

      static void contract(Site<device, Base>& res,
                           Site<device, Base> site1,
                           Site<device, Base> site2,
                           const std::vector<Legs>& legs1,
                           const std::vector<Legs>& legs2,
                           const std::map<Legs, Legs>& map1 = {},
                           const std::map<Legs, Legs>& map2 = {}) {
        // absorb env between two site
        for (const auto& i : site1.neighbor) {
          if (&i.second.site()==&site2) {
            site1.absorb_env(i.first, i.second);
          } // if in
        } // for absorb
        // contract tensor
        Tensor<device, Base> t = Tensor<device, Base>::contract(site1.tensor(), site2.tensor(), legs1, legs2, map1, map2);
        res.neighbor.clear();
        res.set(std::move(t));
        // set edge of new site
        for (const auto& i : site1.neighbor) {
          if (&i.second.site()!=&site2) {
            Legs new_leg = internal::replace_or_not(map1, i.first);
            res(new_leg) = std::move(i.second);
          } // if not connect
        } // for 1
        for (const auto& i : site2.neighbor) {
          if (&i.second.site()!=&site1) {
            Legs new_leg = internal::replace_or_not(map2, i.first);
            res(new_leg) = std::move(i.second);
          } // if not connect
        } // for 2
      } // contract, contract env between 2 site only, without other env linked with them

      static void svd(Site<device, Base>& U, Site<device, Base>& V, Site<device, Base> site,
                      const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs) {
        // absorb all env before svd
        site.absorb_all();
        // svd tensor
        auto tensor_res = site.tensor().svd(input_u_legs, new_u_legs, new_v_legs);
        U.neighbor.clear();
        V.neighbor.clear();
        U.set(std::move(tensor_res.U));
        V.set(std::move(tensor_res.V));
        // set edge
        U.link(new_u_legs, V, new_v_legs);
        V.link(new_v_legs, U, new_u_legs);
        //res.V.unlink_env(legs2, false);
      } // svd

      void qr();

      // high level op

     private:
      void qr_off(const std::vector<Legs>& q_legs, const Legs& leg_q, const Legs& leg_r) {
        auto qr = tensor().qr(q_legs, {leg_q}, {leg_r});
        set(std::move(qr.Q));
      } // qr_off
      void qr_off(const Legs& leg_q, const Legs& leg_r) {
        std::vector<Legs> q_legs = internal::vector_except(tensor().legs, leg_q);
        qr_off(q_legs, leg_q, leg_r);
      } // qr_off
     public:
      void qr_off(const Legs& leg) {
        qr_off(leg, -leg);
      } // qr_off

     private:
      void qr_to(Site<device, Base>& other, const std::vector<Legs>& q_legs, const Legs& leg_q, const Legs& leg_r) {
        auto qr = tensor().qr(q_legs, {leg_q}, {leg_r});
        set(std::move(qr.Q));
        other.set(std::move(other.tensor().contract(qr.R, {leg_r}, {leg_q})));
      } // qr_to
      void qr_to(Site<device, Base>& other, const Legs& leg_q, const Legs& leg_r) {
        std::vector<Legs> q_legs = internal::vector_except(tensor().legs, leg_q);
        qr_to(other, q_legs, leg_q, leg_r);
      } // qr_to
     public:
      void qr_to(Site<device, Base>& other, const Legs& leg) {
        qr_to(other, leg, -leg);
      } // qr_to

     private:
      void update_to(Site<device, Base>& site2,
                     const Legs& leg1, const Legs& leg2,
                     const std::vector<Legs>& tmp_leg1,
                     const Size& D, const Tensor<device, Base>& updater,
                     const std::map<Legs, Legs>& leg_to_tmp1, const std::map<Legs, Legs>& leg_to_tmp2,
                     const std::map<Legs, Legs>& tmp_to_leg1, const std::map<Legs, Legs>& tmp_to_leg2) {
        Site<device, Base>& site1 = *this;
        auto res = site1.tensor()
                   .contract(site2.tensor(), {leg1}, {leg2}, leg_to_tmp1, leg_to_tmp2)
                   .contract(updater, {TAT::Legs::Phy1, TAT::Legs::Phy2}, {TAT::Legs::Phy3, TAT::Legs::Phy4})
                   .svd(tmp_leg1, leg1, leg2, D);
        site1.set(std::move(res.U));
        site1.tensor().legs_rename(tmp_to_leg1);
        site2.set(res.V.multiple(res.S, leg2));
        site2.tensor().legs_rename(tmp_to_leg2);
      } // update
     public:
      void update_to(Site<device, Base>& site2,
                     const Legs& leg1, const Legs& leg2,
                     const Size& D, const Tensor<device, Base>& updater,
                     const std::vector<Legs>& free_leg) {
        using namespace legs_name;
        Site<device, Base>& site1 = *this;
        std::vector<Legs> tmp_leg1 = internal::vector_except(site1.tensor().legs, leg1);
        internal::vector_replace_one(tmp_leg1, Phy, Phy1);
        std::map<Legs, Legs> map1, map2;
        map1[Phy] = Phy2;
        map2[Phy2] = Phy;
        int p = 0;
        for (const auto& i : site2.tensor().legs) {
          if (i!=Phy && i!=leg1 && i!=leg2 && internal::in_vector(i, site1.tensor().legs)) {
            map1[i] = free_leg[p];
            map2[free_leg[p]] = i;
            p++;
          } // if same
        } // for leg
        update_to(site2, leg1, leg2, tmp_leg1, D, updater, {{Phy, Phy1}}, map1, {{Phy1, Phy}}, map2);
      } // update
    }; // class Site
  } // namespace site
} // namespace TAT

#endif // TAT_Site_HPP_
