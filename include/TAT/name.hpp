/**
 * \file name.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <map>
#include <set>
#include <string>

namespace TAT {
   /**
    * \brief 用于给张量的边命名的类型Name, 直接使用字符串
    */
   using SimpleName = std::string;

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
    *
    * \note 这个参数放在Name类外面, 是为了在gdb中显示得比较好看
    */
   inline std::map<std::string, NameIdType> name_to_id = {};
   /**
    * \brief 标号到Name的字符串的映射表
    */
   inline std::map<NameIdType, std::string> id_to_name = {};

   /**
    * \brief 用于给张量的边命名的类型Name, 新建Name的时候可以选定标号, 也可以选定字符串作为名称, Name将自动保证标号和名称的一一对应
    * \note 一个Name拥有一个标号, 而每个标号对应一个双向唯一的字符串作为名字, 有全局变量names_total_index维护目前已分配的标号量,
    * 新建一个字符串的Name时将递增names_total_index并获取一个唯一的标号
    * \see names_total_index
    */
   struct FastName {
      /**
       * \brief Name的标号
       */
      NameIdType id = -1;
      FastName() = default;
      FastName(const NameIdType id) noexcept : id(id) {}
      FastName(const char* name) noexcept : FastName(std::string(name)) {}
      FastName(const std::string& name) noexcept {
         if (const auto position = name_to_id.find(name); position == name_to_id.end()) {
            id = names_total_index++;
            name_to_id[name] = id;
            id_to_name[id] = name;
         } else {
            id = position->second;
         }
      }
      operator const std::string&() const {
         return id_to_name.at(id);
      }
   };

   // 此处将可被c++20的operator<=>替换
   // 生成Name的比较运算符重载
#define TAT_DEFINE_NAME_OPERATOR(OP, EXP)                           \
   inline bool OP(const FastName& name_1, const FastName& name_2) { \
      return EXP;                                                   \
   }
   TAT_DEFINE_NAME_OPERATOR(operator==, name_1.id == name_2.id)
   TAT_DEFINE_NAME_OPERATOR(operator!=, name_1.id != name_2.id)
   TAT_DEFINE_NAME_OPERATOR(operator>=, name_1.id >= name_2.id)
   TAT_DEFINE_NAME_OPERATOR(operator<=, name_1.id <= name_2.id)
   TAT_DEFINE_NAME_OPERATOR(operator>, name_1.id > name_2.id)
   TAT_DEFINE_NAME_OPERATOR(operator<, name_1.id < name_2.id)
#undef TAT_DEFINE_NAME_OPERATOR

   using Name =
#ifdef TAT_USE_SIMPLE_NAME
         SimpleName
#else
         FastName
#endif
         ;

   /**
    * \brief 由名字列表构造名字到序号的映射表
    */
   template<typename NameType>
   std::map<NameType, Rank> construct_name_to_index(const std::vector<NameType>& names) {
      std::map<NameType, Rank> result;
      for (auto name_index = 0; name_index < names.size(); name_index++) {
         result[names[name_index]] = name_index;
      }
      return result;
   }

   /**
    * \brief 判断一个名字列表names是否合法, 即无重复且个数与rank相同
    */
   template<typename NameType>
   bool check_valid_name(const std::vector<NameType>& names, const Rank& rank) {
      const auto result_duplicated = names.size() == std::set<NameType>(names.begin(), names.end()).size();
      const auto result_length = names.size() == rank;
      if (!result_duplicated) {
         TAT_error("Duplicated names in name list");
      }
      if (!result_length) {
         TAT_error("Wrong name list length which no equals to expected length");
      }
      return result_duplicated && result_length;
   }
} // namespace TAT
#endif
