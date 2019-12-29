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
#ifndef TAT_SYMMETRY_HPP_
#   define TAT_SYMMETRY_HPP_

#   include "misc.hpp"

namespace TAT {
   struct NoSymmetry {
      static bool get_parity(
            [[maybe_unused]] const vector<NoSymmetry>& syms,
            [[maybe_unused]] const vector<Rank>& plan) {
         return false;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<NoSymmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         return false;
      }
   };
   NoSymmetry
   operator+([[maybe_unused]] const NoSymmetry& s1, [[maybe_unused]] const NoSymmetry& s2) {
      return NoSymmetry();
   }
   NoSymmetry& operator+=(NoSymmetry& s1, [[maybe_unused]] const NoSymmetry& s2) {
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const NoSymmetry&);
   std::ostream& operator<=(std::ostream& out, const NoSymmetry&);
   std::istream& operator>=(std::istream& in, NoSymmetry&);
#   define TAT_DEF_SYM_OP(OP, EXP)                    \
      bool OP(const NoSymmetry&, const NoSymmetry&) { \
         return EXP;                                  \
      }
   TAT_DEF_SYM_OP(operator==, true)
   TAT_DEF_SYM_OP(operator!=, false)
   TAT_DEF_SYM_OP(operator>=, true)
   TAT_DEF_SYM_OP(operator<=, true)
   TAT_DEF_SYM_OP(operator>, false)
   TAT_DEF_SYM_OP(operator<, false)
#   undef TAT_DEF_SYM_OP

   struct Z2Symmetry {
      Z2 z2 = 0;

      Z2Symmetry(Z2 z2 = 0) : z2(z2) {}

      static bool get_parity(
            [[maybe_unused]] const vector<Z2Symmetry>& syms,
            [[maybe_unused]] const vector<Rank>& plan) {
         return false;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<Z2Symmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         return false;
      }
   };
   Z2Symmetry operator+(const Z2Symmetry& s1, const Z2Symmetry& s2) {
      return Z2Symmetry(s1.z2 ^ s2.z2);
   }
   Z2Symmetry& operator+=(Z2Symmetry& s1, const Z2Symmetry& s2) {
      s1.z2 ^= s2.z2;
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const Z2Symmetry& s);
   std::ostream& operator<=(std::ostream& out, const Z2Symmetry& s);
   std::istream& operator>=(std::istream& in, Z2Symmetry& s);
#   define TAT_DEF_SYM_OP(OP, EXP)                        \
      bool OP(const Z2Symmetry& a, const Z2Symmetry& b) { \
         return EXP;                                      \
      }
   TAT_DEF_SYM_OP(operator==, a.z2 == b.z2)
   TAT_DEF_SYM_OP(operator!=, a.z2 != b.z2)
   TAT_DEF_SYM_OP(operator>=, a.z2 >= b.z2)
   TAT_DEF_SYM_OP(operator<=, a.z2 <= b.z2)
   TAT_DEF_SYM_OP(operator>, a.z2> b.z2)
   TAT_DEF_SYM_OP(operator<, a.z2<b.z2)
#   undef TAT_DEF_SYM_OP

   struct U1Symmetry {
      U1 u1 = 0;

      U1Symmetry(U1 u1 = 0) : u1(u1) {}

      static bool get_parity(
            [[maybe_unused]] const vector<U1Symmetry>& syms,
            [[maybe_unused]] const vector<Rank>& plan) {
         return false;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<U1Symmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         return false;
      }
   };
   U1Symmetry operator+(const U1Symmetry& s1, const U1Symmetry& s2) {
      return U1Symmetry(s1.u1 + s2.u1);
   }
   U1Symmetry& operator+=(U1Symmetry& s1, const U1Symmetry& s2) {
      s1.u1 += s2.u1;
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const U1Symmetry& s);
   std::ostream& operator<=(std::ostream& out, const U1Symmetry& s);
   std::istream& operator>=(std::istream& in, U1Symmetry& s);

#   define TAT_DEF_SYM_OP(OP, EXP)                        \
      bool OP(const U1Symmetry& a, const U1Symmetry& b) { \
         return EXP;                                      \
      }
   TAT_DEF_SYM_OP(operator==, a.u1 == b.u1)
   TAT_DEF_SYM_OP(operator!=, a.u1 != b.u1)
   TAT_DEF_SYM_OP(operator>=, a.u1 >= b.u1)
   TAT_DEF_SYM_OP(operator<=, a.u1 <= b.u1)
   TAT_DEF_SYM_OP(operator>, a.u1> b.u1)
   TAT_DEF_SYM_OP(operator<, a.u1<b.u1)
#   undef TAT_DEF_SYM_OP

   struct FermiSymmetry {
      Fermi fermi = 0;

      FermiSymmetry(Fermi fermi = 0) : fermi(fermi) {}

      static bool get_parity(const vector<FermiSymmetry>& syms, const vector<Rank>& plan) {
         auto rank = Rank(plan.size());
         bool res = false;
         for (Rank i = 0; i < rank; i++) {
            for (Rank j = i + 1; j < rank; j++) {
               if (plan[i] > plan[j]) {
                  res ^= (bool(syms[i].fermi % 2) && bool(syms[j].fermi % 2));
               }
            }
         }
         return res;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<FermiSymmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         bool res = false;
         for (const auto& i : sm_list) {
            Fermi s = 0;
            Fermi s2 = 0;
            for (Rank j = std::get<0>(i); j < std::get<1>(i); j++) {
               Fermi t = syms[j].fermi;
               s += t;
               s2 += t * t;
            }
            res ^= bool(((s * s - s2) / 2) % 2);
         }
         return res;
      }
   };
   FermiSymmetry operator+(const FermiSymmetry& s1, const FermiSymmetry& s2) {
      return FermiSymmetry(s1.fermi + s2.fermi);
   }
   FermiSymmetry& operator+=(FermiSymmetry& s1, const FermiSymmetry& s2) {
      s1.fermi += s2.fermi;
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const FermiSymmetry& s);
   std::ostream& operator<=(std::ostream& out, const FermiSymmetry& s);
   std::istream& operator>=(std::istream& in, FermiSymmetry& s);

#   define TAT_DEF_SYM_OP(OP, EXP)                              \
      bool OP(const FermiSymmetry& a, const FermiSymmetry& b) { \
         return EXP;                                            \
      }
   TAT_DEF_SYM_OP(operator==, a.fermi == b.fermi)
   TAT_DEF_SYM_OP(operator!=, a.fermi != b.fermi)
   TAT_DEF_SYM_OP(operator>=, a.fermi >= b.fermi)
   TAT_DEF_SYM_OP(operator<=, a.fermi <= b.fermi)
   TAT_DEF_SYM_OP(operator>, a.fermi> b.fermi)
   TAT_DEF_SYM_OP(operator<, a.fermi<b.fermi)
#   undef TAT_DEF_SYM_OP

   struct FermiZ2Symmetry {
      Fermi fermi = 0;
      Z2 z2 = 0;

      FermiZ2Symmetry(Fermi fermi = 0, Z2 z2 = 0) : fermi(fermi), z2(z2) {}

      static bool get_parity(const vector<FermiZ2Symmetry>& syms, const vector<Rank>& plan) {
         auto rank = Rank(plan.size());
         bool res = false;
         for (Rank i = 0; i < rank; i++) {
            for (Rank j = i + 1; j < rank; j++) {
               if (plan[i] > plan[j]) {
                  res ^= (bool(syms[i].fermi % 2) && bool(syms[j].fermi % 2));
               }
            }
         }
         return res;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<FermiZ2Symmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         bool res = false;
         for (const auto& i : sm_list) {
            Fermi s = 0;
            Fermi s2 = 0;
            for (Rank j = std::get<0>(i); j < std::get<1>(i); j++) {
               Fermi t = syms[j].fermi;
               s += t;
               s2 += t * t;
            }
            res ^= bool(((s * s - s2) / 2) % 2);
         }
         return res;
      }
   };
   FermiZ2Symmetry operator+(const FermiZ2Symmetry& s1, const FermiZ2Symmetry& s2) {
      return FermiZ2Symmetry(s1.fermi + s2.fermi, s1.z2 ^ s2.z2);
   }
   FermiZ2Symmetry& operator+=(FermiZ2Symmetry& s1, const FermiZ2Symmetry& s2) {
      s1.fermi += s2.fermi;
      s1.z2 ^= s2.z2;
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const FermiZ2Symmetry& s);
   std::ostream& operator<=(std::ostream& out, const FermiZ2Symmetry& s);
   std::istream& operator>=(std::istream& in, FermiZ2Symmetry& s);

#   define TAT_DEF_SYM_OP(OP, EXP)                                  \
      bool OP(const FermiZ2Symmetry& a, const FermiZ2Symmetry& b) { \
         return EXP;                                                \
      }
   TAT_DEF_SYM_OP(operator==,(a.fermi == b.fermi) && (a.z2 == b.z2))
   TAT_DEF_SYM_OP(operator!=,(a.fermi != b.fermi) || (a.z2 != b.z2))
   TAT_DEF_SYM_OP(operator>=,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.z2 >= b.z2)))
   TAT_DEF_SYM_OP(operator<=,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.z2 <= b.z2)))
   TAT_DEF_SYM_OP(operator>,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.z2 > b.z2)))
   TAT_DEF_SYM_OP(operator<,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.z2 < b.z2)))
#   undef TAT_DEF_SYM_OP

   struct FermiU1Symmetry {
      Fermi fermi = 0;
      U1 u1 = 0;

      FermiU1Symmetry(Fermi fermi = 0, U1 u1 = 0) : fermi(fermi), u1(u1) {}

      static bool get_parity(const vector<FermiU1Symmetry>& syms, const vector<Rank>& plan) {
         auto rank = Rank(plan.size());
         bool res = false;
         for (Rank i = 0; i < rank; i++) {
            for (Rank j = i + 1; j < rank; j++) {
               if (plan[i] > plan[j]) {
                  res ^= (bool(syms[i].fermi % 2) && bool(syms[j].fermi % 2));
               }
            }
         }
         return res;
      }
      static bool get_parity(
            [[maybe_unused]] const vector<FermiU1Symmetry>& syms,
            [[maybe_unused]] const vector<std::tuple<Rank, Rank>>& sm_list) {
         bool res = false;
         for (const auto& i : sm_list) {
            Fermi s = 0;
            Fermi s2 = 0;
            for (Rank j = std::get<0>(i); j < std::get<1>(i); j++) {
               Fermi t = syms[j].fermi;
               s += t;
               s2 += t * t;
            }
            res ^= bool(((s * s - s2) / 2) % 2);
         }
         return res;
      }
   };
   FermiU1Symmetry operator+(const FermiU1Symmetry& s1, const FermiU1Symmetry& s2) {
      return FermiU1Symmetry(s1.fermi + s2.fermi, s1.u1 ^ s2.u1);
   }
   FermiU1Symmetry& operator+=(FermiU1Symmetry& s1, const FermiU1Symmetry& s2) {
      s1.fermi += s2.fermi;
      s1.u1 ^= s2.u1;
      return s1;
   }

   std::ostream& operator<<(std::ostream& out, const FermiU1Symmetry& s);
   std::ostream& operator<=(std::ostream& out, const FermiU1Symmetry& s);
   std::istream& operator>=(std::istream& in, FermiU1Symmetry& s);

#   define TAT_DEF_SYM_OP(OP, EXP)                                  \
      bool OP(const FermiU1Symmetry& a, const FermiU1Symmetry& b) { \
         return EXP;                                                \
      }
   TAT_DEF_SYM_OP(operator==,(a.fermi == b.fermi) && (a.u1 == b.u1))
   TAT_DEF_SYM_OP(operator!=,(a.fermi != b.fermi) || (a.u1 != b.u1))
   TAT_DEF_SYM_OP(operator>=,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.u1 >= b.u1)))
   TAT_DEF_SYM_OP(operator<=,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.u1 <= b.u1)))
   TAT_DEF_SYM_OP(operator>,(a.fermi > b.fermi) || ((a.fermi == b.fermi) && (a.u1 > b.u1)))
   TAT_DEF_SYM_OP(operator<,(a.fermi < b.fermi) || ((a.fermi == b.fermi) && (a.u1 < b.u1)))
#   undef TAT_DEF_SYM_OP

} // namespace TAT
#endif
