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

   struct fast_name_dataset_t {
      /**
       * \brief Name中用于标号的类型
       */
      using FastNameId = int;

      /**
       * \brief Name的全局计数, 每当新建一个Name都会是指递增并获取一个关于Name的字符串唯一的标号
       */
      FastNameId names_total_index = 1;

      /**
       * \brief Name的字符串到标号的映射表
       *
       * \note 这个参数放在Name类外面, 是为了在gdb中显示得比较好看
       */
      std::map<std::string, FastNameId> name_to_id = {{"", 0}};
      /**
       * \brief 标号到Name的字符串的映射表
       */
      std::vector<std::string> id_to_name = {""};
   };

   inline auto fast_name_dataset = fast_name_dataset_t();

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
      fast_name_dataset_t::FastNameId id = 0; // 默认为空串, 行为和std::string一致
      FastName() = default;
      FastName(const fast_name_dataset_t::FastNameId id) noexcept : id(id) {}
      FastName(const char* name) noexcept : FastName(std::string(name)) {}
      FastName(const std::string& name) noexcept {
         if (const auto position = fast_name_dataset.name_to_id.find(name); position == fast_name_dataset.name_to_id.end()) {
            id = fast_name_dataset.name_to_id[name] = fast_name_dataset.names_total_index++;
            fast_name_dataset.id_to_name.push_back(name);
         } else {
            id = position->second;
         }
      }
      operator const std::string&() const {
         return fast_name_dataset.id_to_name[id];
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

   using DefaultName =
#ifdef TAT_USE_SIMPLE_NAME
         SimpleName
#else
         FastName
#endif
         ;

   // 作为Name需要可以比较, 需要text/binary的io供输入输出
   // 在python/TAT.cpp中还需要到std::string的转换函数
   // 最后需要通过下面的NameTraits设置内部名称
   template<typename Name>
   struct NameTraits {
      static const Name Contract_0;
      static const Name Contract_1;
      static const Name Contract_2;
      static const Name SVD_U;
      static const Name SVD_V;
      static const Name QR_1;
      static const Name QR_2;
      static const Name Trace_1;
      static const Name Trace_2;
      static const Name Trace_3;
      static const Name No_Old_Name;
      static const Name No_New_Name;
      static const Name Internal_0;
      static const Name Internal_1;
      static const Name Internal_2;
   };
#define TAT_DEFINE_DEFAULT_INTERNAL_NAME(x, n) \
   template<typename Name>                     \
   const Name NameTraits<Name>::x = NameTraits<Name>::Internal_##n;
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_0, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_U, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_V, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_1, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_2, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_3, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_Old_Name, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_New_Name, 0)
#undef TAT_DEFINE_DEFAULT_INTERNAL_NAME
#define TAT_DEFINE_INTERNAL_NAME(x)                      \
   template<>                                            \
   const std::string NameTraits<std::string>::x("," #x); \
   template<>                                            \
   const FastName NameTraits<FastName>::x("," #x);
   TAT_DEFINE_INTERNAL_NAME(Contract_0)
   TAT_DEFINE_INTERNAL_NAME(Contract_1)
   TAT_DEFINE_INTERNAL_NAME(Contract_2)
   TAT_DEFINE_INTERNAL_NAME(SVD_U)
   TAT_DEFINE_INTERNAL_NAME(SVD_V)
   TAT_DEFINE_INTERNAL_NAME(QR_1)
   TAT_DEFINE_INTERNAL_NAME(QR_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_1)
   TAT_DEFINE_INTERNAL_NAME(Trace_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_3)
   TAT_DEFINE_INTERNAL_NAME(No_Old_Name)
   TAT_DEFINE_INTERNAL_NAME(No_New_Name)
#undef TAT_DEFINE_INTERNAL_NAME

   /**
    * \brief 由名字列表构造名字到序号的映射表
    */
   template<typename Name>
   std::map<Name, Rank> construct_name_to_index(const std::vector<Name>& names) {
      std::map<Name, Rank> result;
      for (auto name_index = 0; name_index < names.size(); name_index++) {
         result[names[name_index]] = name_index;
      }
      return result;
   }

   /**
    * \brief 判断一个名字列表names是否合法, 即无重复且个数与rank相同
    */
   template<typename Name>
   bool check_valid_name(const std::vector<Name>& names, const Rank& rank) {
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
