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
#ifdef TAT_USE_SIMPLE_NAME
   /**
    * \brief 用于给张量的边命名的类型Name, 直接使用字符串
    */
   struct Name {
      std::string name;
      Name(const std::string& name) noexcept : name(name) {}
      Name(const char* name) : name(name) {}
      Name() : name("Null") {}

      [[nodiscard]] const std::string& get_name() const {
         return name;
      }
   };
#define TAT_NAME_KEY name
#else
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
   struct Name {
      /**
       * \brief Name的标号
       */
      NameIdType id = -1;
      Name() = default;
      Name(const NameIdType id) noexcept : id(id) {}
      Name(const char* name) noexcept : Name(std::string(name)) {}
      Name(const std::string& name) noexcept {
         if (const auto position = name_to_id.find(name); position == name_to_id.end()) {
            id = names_total_index++;
            name_to_id[name] = id;
            id_to_name[id] = name;
         } else {
            id = position->second;
         }
      }

      [[nodiscard]] const std::string& get_name() const {
         return id_to_name.at(id);
      }
   };
#define TAT_NAME_KEY id
#endif

   // 此处将可被c++20的operator<=>替换
   // 生成Name的比较运算符重载
#define TAT_DEFINE_NAME_OPERATOR(OP, EXP)                   \
   inline bool OP(const Name& name_1, const Name& name_2) { \
      return EXP;                                           \
   }
   TAT_DEFINE_NAME_OPERATOR(operator==, name_1.TAT_NAME_KEY == name_2.TAT_NAME_KEY)
   TAT_DEFINE_NAME_OPERATOR(operator!=, name_1.TAT_NAME_KEY != name_2.TAT_NAME_KEY)
   TAT_DEFINE_NAME_OPERATOR(operator>=, name_1.TAT_NAME_KEY >= name_2.TAT_NAME_KEY)
   TAT_DEFINE_NAME_OPERATOR(operator<=, name_1.TAT_NAME_KEY <= name_2.TAT_NAME_KEY)
   TAT_DEFINE_NAME_OPERATOR(operator>, name_1.TAT_NAME_KEY > name_2.TAT_NAME_KEY)
   TAT_DEFINE_NAME_OPERATOR(operator<, name_1.TAT_NAME_KEY < name_2.TAT_NAME_KEY)
#undef TAT_DEFINE_NAME_OPERATOR
#undef TAT_NAME_KEY

   // 保留名称, 在一些张量运算内部使用
#define TAT_DEFINE_INTERNAL_NAME(x) inline const Name x(#x)
   namespace internal_name {
      TAT_DEFINE_INTERNAL_NAME(Null);
      TAT_DEFINE_INTERNAL_NAME(Contract_0);
      TAT_DEFINE_INTERNAL_NAME(Contract_1);
      TAT_DEFINE_INTERNAL_NAME(Contract_2);
      TAT_DEFINE_INTERNAL_NAME(SVD_U);
      TAT_DEFINE_INTERNAL_NAME(SVD_V);
      TAT_DEFINE_INTERNAL_NAME(QR_1);
      TAT_DEFINE_INTERNAL_NAME(QR_2);
      TAT_DEFINE_INTERNAL_NAME(Trace_1);
      TAT_DEFINE_INTERNAL_NAME(Trace_2);
      TAT_DEFINE_INTERNAL_NAME(Trace_3);
      TAT_DEFINE_INTERNAL_NAME(U_edge);
      TAT_DEFINE_INTERNAL_NAME(V_edge);
   } // namespace internal_name
   namespace common_name {
#define TAT_DEFINE_COMMON_NAME_WITH_INDEX(x) \
   TAT_DEFINE_INTERNAL_NAME(x);              \
   TAT_DEFINE_INTERNAL_NAME(x##0);           \
   TAT_DEFINE_INTERNAL_NAME(x##1);           \
   TAT_DEFINE_INTERNAL_NAME(x##2);           \
   TAT_DEFINE_INTERNAL_NAME(x##3);           \
   TAT_DEFINE_INTERNAL_NAME(x##4);           \
   TAT_DEFINE_INTERNAL_NAME(x##5);           \
   TAT_DEFINE_INTERNAL_NAME(x##6);           \
   TAT_DEFINE_INTERNAL_NAME(x##7);           \
   TAT_DEFINE_INTERNAL_NAME(x##8);           \
   TAT_DEFINE_INTERNAL_NAME(x##9);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(P);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Phy);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(L);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Left);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(R);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Right);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(U);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Up);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(D);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Down);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(I);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(In);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(O);
      TAT_DEFINE_COMMON_NAME_WITH_INDEX(Out);
#undef TAT_DEFINE_COMMON_NAME_WITH_INDEX
   } // namespace common_name
#undef TAT_DEFINE_INTERNAL_NAME

   /**
    * \brief 由名字列表构造名字到序号的映射表
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
      const auto result_duplicated = names.size() == std::set<Name>(names.begin(), names.end()).size();
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
