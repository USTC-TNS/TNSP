/**
 * \file name.hpp
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
#ifndef TAT_NAME_HPP_
#   define TAT_NAME_HPP_

namespace TAT {
   using NameIdType = int;

   NameIdType names_total = 0;
   std::map<std::string, NameIdType> name_to_id = {};
   std::map<NameIdType, std::string> id_to_name = {};

   struct Name {
      NameIdType id = -1;
      Name() = default;
      Name(NameIdType id) : id{id} {}
      Name(const char* s) : Name(std::string(s)) {}
      Name(const std::string& name) {
         auto pos = name_to_id.find(name);
         if (pos == name_to_id.end()) {
            id = names_total++;
            name_to_id[name] = id;
            id_to_name[id] = name;
         } else {
            id = pos->second;
         }
      }
   };

   std::ostream& operator<<(std::ostream& out, const Name& name);
   std::ostream& operator<=(std::ostream& out, const Name& name);
   std::istream& operator>=(std::istream& in, Name& name);

#   define TAT_DEF_NAME_OP(OP, EXP)           \
      bool OP(const Name& a, const Name& b) { \
         return EXP;                          \
      }
   TAT_DEF_NAME_OP(operator==, a.id == b.id)
   TAT_DEF_NAME_OP(operator!=, a.id != b.id)
   TAT_DEF_NAME_OP(operator>=, a.id >= b.id)
   TAT_DEF_NAME_OP(operator<=, a.id <= b.id)
   TAT_DEF_NAME_OP(operator>, a.id> b.id)
   TAT_DEF_NAME_OP(operator<, a.id<b.id)
#   undef TAT_DEF_NAME_OP

#   define TAT_DEF_NAME(x) const Name x(#   x)
#   define TAT_DEF_NAMES(n)      \
      TAT_DEF_NAME(Phy##n);      \
      TAT_DEF_NAME(Left##n);     \
      TAT_DEF_NAME(Right##n);    \
      TAT_DEF_NAME(Up##n);       \
      TAT_DEF_NAME(Down##n);     \
      TAT_DEF_NAME(LeftUp##n);   \
      TAT_DEF_NAME(LeftDown##n); \
      TAT_DEF_NAME(RightUp##n);  \
      TAT_DEF_NAME(RightDown##n)
   TAT_DEF_NAMES();
   TAT_DEF_NAMES(1);
   TAT_DEF_NAMES(2);
   TAT_DEF_NAMES(3);
   TAT_DEF_NAMES(4);
#   undef TAT_DEF_NAMES
   TAT_DEF_NAME(Contract1);
   TAT_DEF_NAME(Contract2);
#   undef TAT_DEF_NAME
} // namespace TAT
#endif
