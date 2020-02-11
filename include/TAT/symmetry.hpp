/**
 * \file symmetry.hpp
 *
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

#pragma once
#ifndef TAT_SYMMETRY_HPP
#define TAT_SYMMETRY_HPP

#include "misc.hpp"

namespace TAT {
   // TODO: 关于对称性的get parity需要完善接口， 以及需要注释
   // 因为arrow的原因， symmetry可能需要operator-=等符号
   template<class Derived>
   struct bose_symmetry : bose_symmetry_base {};

   template<class Derived>
   struct fermi_symmetry : fermi_symmetry_base {
      static bool get_reverse_parity(const vector<Derived>& symmetries, const vector<bool>& flag) {
         auto res = false;
         for (auto i = 0; i < flag.size(); i++) {
            if (flag[i]) {
               res ^= bool(symmetries[i].fermi % 2);
            }
         }
         return res;
      }
      static bool
      get_transpose_parity(const vector<Derived>& symmetries, const vector<Rank>& plan) {
         auto res = false;
         for (auto i = 0; i < plan.size(); i++) {
            for (auto j = i + 1; j < plan.size(); j++) {
               if (plan[i] > plan[j]) {
                  res ^= (bool(symmetries[i].fermi % 2) && bool(symmetries[j].fermi % 2));
               }
            }
         }
         return res;
      }
      static bool get_split_merge_parity(
            [[maybe_unused]] const vector<Derived>& symmetries,
            [[maybe_unused]] const vector<Rank>& flag) {
         auto res = false;
         auto s = 0;
         auto s2 = 0;
         auto tmp = 0;
         for (auto i = 0; i < flag.size(); i++) {
            if (tmp != flag[i]) {
               res ^= bool(((s * s - s2) / 2) % 2);
               s = 0;
               s2 = 0;
               tmp++;
            }
            const auto t = symmetries[i].fermi;
            s += t;
            s2 += t * t;
         }
         return res;
      }
   };

   struct NoSymmetry : bose_symmetry<NoSymmetry> {};
   inline NoSymmetry
   operator+([[maybe_unused]] const NoSymmetry& s1, [[maybe_unused]] const NoSymmetry& s2) {
      return NoSymmetry();
   }
   inline NoSymmetry& operator+=(NoSymmetry& s1, [[maybe_unused]] const NoSymmetry& s2) {
      return s1;
   }

   inline std::ostream& operator<<(std::ostream& out, const NoSymmetry&);
#define TAT_DEF_SYM_OP(OP, EXP)                           \
   inline bool OP(const NoSymmetry&, const NoSymmetry&) { \
      return EXP;                                         \
   }
   TAT_DEF_SYM_OP(operator==, true)
   TAT_DEF_SYM_OP(operator!=, false)
   TAT_DEF_SYM_OP(operator>=, true)
   TAT_DEF_SYM_OP(operator<=, true)
   TAT_DEF_SYM_OP(operator>, false)
   TAT_DEF_SYM_OP(operator<, false)
#undef TAT_DEF_SYM_OP

   struct Z2Symmetry : bose_symmetry<Z2Symmetry> {
      Z2 z2;

      Z2Symmetry(const Z2 z2 = false) : z2(z2) {}
   };
   inline Z2Symmetry operator+(const Z2Symmetry& s1, const Z2Symmetry& s2) {
      return Z2Symmetry(s1.z2 ^ s2.z2);
   }
   inline Z2Symmetry& operator+=(Z2Symmetry& s1, const Z2Symmetry& s2) {
      s1.z2 ^= s2.z2;
      return s1;
   }

   inline std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                               \
   inline bool OP(const Z2Symmetry& a, const Z2Symmetry& b) { \
      return EXP;                                             \
   }
   TAT_DEF_SYM_OP(operator==, a.z2 == b.z2)
   TAT_DEF_SYM_OP(operator!=, a.z2 != b.z2)
   TAT_DEF_SYM_OP(operator>=, a.z2 >= b.z2)
   TAT_DEF_SYM_OP(operator<=, a.z2 <= b.z2)
   TAT_DEF_SYM_OP(operator>, a.z2> b.z2)
   TAT_DEF_SYM_OP(operator<, a.z2<b.z2)
#undef TAT_DEF_SYM_OP

   struct U1Symmetry : bose_symmetry<U1Symmetry> {
      U1 u1;

      U1Symmetry(const U1 u1 = 0) : u1(u1) {}
   };
   inline U1Symmetry operator+(const U1Symmetry& s1, const U1Symmetry& s2) {
      return U1Symmetry(s1.u1 + s2.u1);
   }
   inline U1Symmetry& operator+=(U1Symmetry& s1, const U1Symmetry& s2) {
      s1.u1 += s2.u1;
      return s1;
   }

   inline std::ostream& operator<<(std::ostream& out, const U1Symmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                               \
   inline bool OP(const U1Symmetry& a, const U1Symmetry& b) { \
      return EXP;                                             \
   }
   TAT_DEF_SYM_OP(operator==, a.u1 == b.u1)
   TAT_DEF_SYM_OP(operator!=, a.u1 != b.u1)
   TAT_DEF_SYM_OP(operator>=, a.u1 >= b.u1)
   TAT_DEF_SYM_OP(operator<=, a.u1 <= b.u1)
   TAT_DEF_SYM_OP(operator>, a.u1> b.u1)
   TAT_DEF_SYM_OP(operator<, a.u1<b.u1)
#undef TAT_DEF_SYM_OP

   struct FermiSymmetry : fermi_symmetry<FermiSymmetry> {
      Fermi fermi;

      FermiSymmetry(const Fermi fermi = 0) : fermi(fermi) {}
   };
   inline FermiSymmetry operator+(const FermiSymmetry& s1, const FermiSymmetry& s2) {
      return FermiSymmetry(s1.fermi + s2.fermi);
   }
   inline FermiSymmetry& operator+=(FermiSymmetry& s1, const FermiSymmetry& s2) {
      s1.fermi += s2.fermi;
      return s1;
   }
   inline FermiSymmetry operator!(const FermiSymmetry& s) {
      return FermiSymmetry(-s.fermi);
   }

   inline std::ostream& operator<<(std::ostream& out, const FermiSymmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                                     \
   inline bool OP(const FermiSymmetry& a, const FermiSymmetry& b) { \
      return EXP;                                                   \
   }
   TAT_DEF_SYM_OP(operator==, a.fermi == b.fermi)
   TAT_DEF_SYM_OP(operator!=, a.fermi != b.fermi)
   TAT_DEF_SYM_OP(operator>=, a.fermi >= b.fermi)
   TAT_DEF_SYM_OP(operator<=, a.fermi <= b.fermi)
   TAT_DEF_SYM_OP(operator>, a.fermi> b.fermi)
   TAT_DEF_SYM_OP(operator<, a.fermi<b.fermi)
#undef TAT_DEF_SYM_OP

   struct FermiZ2Symmetry : fermi_symmetry<FermiZ2Symmetry> {
      Fermi fermi;
      Z2 z2;

      FermiZ2Symmetry(const Fermi fermi = 0, const Z2 z2 = false) : fermi(fermi), z2(z2) {}
   };
   inline FermiZ2Symmetry operator+(const FermiZ2Symmetry& s1, const FermiZ2Symmetry& s2) {
      return FermiZ2Symmetry(s1.fermi + s2.fermi, s1.z2 ^ s2.z2);
   }
   inline FermiZ2Symmetry& operator+=(FermiZ2Symmetry& s1, const FermiZ2Symmetry& s2) {
      s1.fermi += s2.fermi;
      s1.z2 ^= s2.z2;
      return s1;
   }
   inline FermiZ2Symmetry operator!(const FermiZ2Symmetry& s) {
      return FermiZ2Symmetry(-s.fermi, s.z2);
   }

   inline std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                                         \
   inline bool OP(const FermiZ2Symmetry& a, const FermiZ2Symmetry& b) { \
      return EXP;                                                       \
   }
   TAT_DEF_SYM_OP(operator==,(a.fermi == b.fermi) && (a.z2 == b.z2))
   TAT_DEF_SYM_OP(operator!=,(a.fermi != b.fermi) || (a.z2 != b.z2))
   TAT_DEF_SYM_OP(operator>=,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.z2 >= b.z2)))
   TAT_DEF_SYM_OP(operator<=,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.z2 <= b.z2)))
   TAT_DEF_SYM_OP(operator>,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.z2 > b.z2)))
   TAT_DEF_SYM_OP(operator<,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.z2 < b.z2)))
#undef TAT_DEF_SYM_OP

   struct FermiU1Symmetry : fermi_symmetry<FermiU1Symmetry> {
      Fermi fermi;
      U1 u1;

      FermiU1Symmetry(const Fermi fermi = 0, const U1 u1 = 0) : fermi(fermi), u1(u1) {}
   };
   inline FermiU1Symmetry operator+(const FermiU1Symmetry& s1, const FermiU1Symmetry& s2) {
      return FermiU1Symmetry(s1.fermi + s2.fermi, s1.u1 + s2.u1);
   }
   inline FermiU1Symmetry& operator+=(FermiU1Symmetry& s1, const FermiU1Symmetry& s2) {
      s1.fermi += s2.fermi;
      s1.u1 += s2.u1;
      return s1;
   }
   inline FermiU1Symmetry operator!(const FermiU1Symmetry& s) {
      return FermiU1Symmetry(-s.fermi, s.u1);
   }

   inline std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& s);
#define TAT_DEF_SYM_OP(OP, EXP)                                         \
   inline bool OP(const FermiU1Symmetry& a, const FermiU1Symmetry& b) { \
      return EXP;                                                       \
   }
   TAT_DEF_SYM_OP(operator==,(a.fermi == b.fermi) && (a.u1 == b.u1))
   TAT_DEF_SYM_OP(operator!=,(a.fermi != b.fermi) || (a.u1 != b.u1))
   TAT_DEF_SYM_OP(operator>=,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.u1 >= b.u1)))
   TAT_DEF_SYM_OP(operator<=,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.u1 <= b.u1)))
   TAT_DEF_SYM_OP(operator>,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.u1 > b.u1)))
   TAT_DEF_SYM_OP(operator<,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.u1 < b.u1)))
#undef TAT_DEF_SYM_OP
} // namespace TAT
#endif
