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

        Edge(const Site<device, Base>& site_ref, const Legs& _legs) : _site(&site_ref), legs(_legs) {}
        Edge() = default;

        Site<device, Base>& site() const {
          return const_cast<Site<device, Base>&>(*_site);
        } // site
        Tensor<device, Base>& env() const {
          return const_cast<Tensor<device, Base>&>(*_env.get());
        } // env

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

      Tensor<device, Base>& tensor() const {
        return const_cast<Tensor<device, Base>&>(*_tensor.get());
      } // tensor
      const Edge& operator()(Legs legs) const {
        return neighbor[legs];
      } // operator()
      Edge& operator()(Legs legs) {
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

      // link
      friend class link_res1;
      class link_res1 {
       public:
        Edge& edge;
        Size size;
      }; // class link_res
      link_res1 link(const Legs& legs1, const Site<device, Base>& site2, const Legs& legs2) {
        Site<device, Base>& site1 = *this;
        Edge& edge_ref = site1(legs1) = Edge(site2, legs2);
        auto pos1 = std::find(site1.tensor().legs.begin(), site1.tensor().legs.end(), legs1);
        assert(pos1!=site1.tensor().legs.end());
        Rank index1 = std::distance(site1.tensor().legs.begin(), pos1);
        return link_res1{edge_ref, site1.tensor().dims()[index1]};
      } // link, single link, return dim

      friend class link_res2;
      class link_res2 {
       public:
        Edge& edge1;
        Edge& edge2;
        Size size;
      }; // class link_res
      static link_res2 link(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        auto res1 = site1.link(legs1, site2, legs2);
        auto res2 = site2.link(legs2, site1, legs1);
        assert(res1.size==res2.size);
        return link_res2{res1.edge, res2.edge, res1.size};
      } // link, double link, return dim

      void link_env(const Legs& legs1, Site<device, Base>& site2, const Legs& legs2, std::shared_ptr<const Tensor<device, Base>> env=std::shared_ptr<const Tensor<device, Base>>()) {
        Site<device, Base>& site1 = *this;
        auto res = link(site1, legs1, site2, legs2);
        Size dim = res.size;
        if (!env) {
          env = std::shared_ptr<const Tensor<device, Base>>(new Tensor<device, Base>({Legs::Phy}, {dim}));
          const_cast<Tensor<device, Base>&>(*env).set_constant(1);
        } else {
          assert(env->dims().size()==1);
          assert(dim==env->size());
          site1.set(std::move(site1.tensor().multiple(1/(*env), legs1)));
        } // if
        res.edge1.set(env);
        res.edge2.set(env);
      } // link_env, double link, insert env

      Edge unlink(const Legs& legs1) {
        Site<device, Base>& site1 = *this;
        auto pos1 = site1.neighbor.find(legs1);
        auto edge = std::move(pos1->second);
        site1.neighbor.erase(pos1);
        return std::move(edge);
      } // unlink, single unlink

      static std::shared_ptr<const Tensor<device, Base>> unlink(Site<device, Base>& site1, const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        auto edge1 = site1.unlink(legs1);
        auto edge2 = site2.unlink(legs2);
        assert(&edge1.site()==&site2);
        assert(&edge2.site()==&site1);
        return edge1._env;
      } // unlink, double unlink

      void unlink_env(const Legs& legs1, Site<device, Base>& site2, const Legs& legs2) {
        Site<device, Base>& site1 = *this;
        auto tmp = unlink(site1, legs1, site2, legs2);
        if (tmp) {
          site1.set(std::move(site1.tensor().multiple(*tmp, legs1)));
        } // if
      } // unlink, double unlink, delete env

      // io
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

      // norm
      template<int n>
      Site<device, Base>& normalize() {
        tensor() /= tensor().template norm<n>();
        return *this;
      } // normalize

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
