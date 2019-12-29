#pragma once
#ifndef TAT_EDGE_HPP_
#   define TAT_EDGE_HPP_

#   include "misc.hpp"

namespace TAT {
   template<class Symmetry>
   struct Edge : public std::map<Symmetry, Size> {
      using std::map<Symmetry, Size>::map;

      Edge(Size s) : std::map<Symmetry, Size>({{Symmetry(), s}}) {}
   };

   template<class Symmetry>
   struct EdgePointer {
      const Edge<Symmetry>* ptr;
      EdgePointer(const Edge<Symmetry>* ptr) : ptr(ptr) {}
      using const_iterator = typename Edge<Symmetry>::const_iterator;
      const_iterator begin() const {
         return ptr->begin();
      }
      const_iterator end() const {
         return ptr->end();
      }
      auto size() const {
         return ptr->size();
      }
   };
   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const EdgePointer<Symmetry>& ptr) {
      return out << *ptr.ptr;
   }

   template<class Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge);
   template<class Symmetry>
   std::ostream& operator<=(std::ostream& out, const Edge<Symmetry>& edge);
   template<class Symmetry>
   std::istream& operator>=(std::istream& in, Edge<Symmetry>& edge);

   template<class T, class F1, class F2, class F3, class F4, class F5>
   void loop_edge(const T& edges, F1&& rank0, F2&& init, F3&& check, F4&& append, F5&& update) {
      auto rank = Rank(edges.size());
      if (!rank) {
         rank0();
         return;
      }
      auto pos = vector<typename T::value_type::const_iterator>();
      for (const auto& i : edges) {
         auto ptr = i.begin();
         if (ptr == i.end()) {
            return;
         }
         pos.push_back(ptr);
      }
      init(pos);
      while (true) {
         if (check(pos)) {
            append(pos);
         }
         Rank ptr = rank-1;
         pos[ptr]++;
         while (pos[ptr] == edges[ptr].end()) {
            if (ptr == 0) {
               return;
            }
            pos[ptr] = edges[ptr].begin();
            ptr--;
            pos[ptr]++;
         }
         update(pos, ptr);
      }
   }

   template<class Symmetry>
   struct EdgePosition {
      Symmetry sym;
      Size position;

      EdgePosition(Size p) : sym(Symmetry()), position(p) {}
      EdgePosition(Symmetry s, Size p) : sym(s), position(p) {}
   };
} // namespace TAT
#endif
