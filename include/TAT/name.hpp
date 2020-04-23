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
#ifndef TAT_NAME_HPP
#define TAT_NAME_HPP

#include "misc.hpp"

#include <map>
#include <set>
#include <string>

namespace TAT {
   /**
    * \brief Name中用于标号的类型
    */
   using NameIdType = int;

   /**
    * \brief Name的全局计数, 每当新建一个Name都会是指递增并获取一个关于Name的字符串唯一的标号
    */
   inline NameIdType names_total_index = 0;
   /**
    * \brief Name的字符串到标号的映射表
    */
   inline std::map<std::string, NameIdType> name_to_id = {};
   /**
    * \brief 标号到Name的字符串的映射表
    */
   inline std::map<NameIdType, std::string> id_to_name = {};

   /**
    * \brief 用于给张量的脚标命名的类型Name, 新建Name的时候可以选定标号, 也可以选定字符串作为名称, Name将自动保证标号和名称的一一对应
    * \note 一个Name拥有一个标号, 而每个标号对应一个双向唯一的字符串作为名字, 有全局变量names_total维护目前已分配的标号量,
    * 新建一个字符串的Name时将递增names_total并获取一个唯一的标号
    * \see names_total_index
    */
   struct Name {
      /**
       * \brief Name的标号
       */
      NameIdType id = -1;
      Name() = default;
      Name(const NameIdType id) : id(id) {}
      Name(const char* name) : Name(std::string(name)) {}
      Name(const std::string& name) {
         if (const auto position = name_to_id.find(name); position == name_to_id.end()) {
            id = names_total_index++;
            name_to_id[name] = id;
            id_to_name[id] = name;
         } else {
            id = position->second;
         }
      }
   };

   // 此处将可被c++20的operator<=>替换
#define TAT_DEF_NAME_OP(OP, EXP)                            \
   inline bool OP(const Name& name_1, const Name& name_2) { \
      return EXP;                                           \
   }
   TAT_DEF_NAME_OP(operator==, name_1.id == name_2.id)
   TAT_DEF_NAME_OP(operator!=, name_1.id != name_2.id)
   TAT_DEF_NAME_OP(operator>=, name_1.id >= name_2.id)
   TAT_DEF_NAME_OP(operator<=, name_1.id <= name_2.id)
   TAT_DEF_NAME_OP(operator>, name_1.id> name_2.id)
   TAT_DEF_NAME_OP(operator<, name_1.id<name_2.id)
#undef TAT_DEF_NAME_OP

   // 保留名称
#define TAT_DEF_NAME(x) inline const Name x(#x)
   TAT_DEF_NAME(Contract1);
   TAT_DEF_NAME(Contract2);
   TAT_DEF_NAME(SVD1);
   TAT_DEF_NAME(SVD2);
#undef TAT_DEF_NAME

   /**
    * \brief 由名字列表构造名字到需要的映射表
    */
   inline std::map<Name, Rank> construct_name_to_index(const std::vector<Name>& names) {
      std::map<Name, Rank> result;
      for (auto name_index = 0; name_index < names.size(); name_index++) {
         result[names[name_index]] = name_index;
      }
      return result;
   }

   /**
    * \brief 判断一个名字列表names是否合法, 即无重复且个数与rank相同
    */
   inline bool check_valid_name(const std::vector<Name>& names, const Rank& rank) {
      auto result = names.size() == std::set<Name>(names.begin(), names.end()).size() && names.size() == rank;
      if (!result) {
         warning_or_error("Wrong Name Number or Duplicated Name");
      }
      return result;
   }
} // namespace TAT
#endif
