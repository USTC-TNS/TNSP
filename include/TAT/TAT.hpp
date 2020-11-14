/**
 * \file TAT.hpp
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
#ifndef TAT_HPP
#define TAT_HPP

#ifndef __cplusplus
#error only work for c++
#endif

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error require c++17 or later
#endif
#else
#if __cplusplus < 201703L
#error require c++17 or later
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// 开关说明
// TAT_USE_MPI 定义以开启MPI支持, cmake可对此进行定义
// TAT_USE_MKL_TRANSPOSE 定义以使用mkl加速转置, cmake可对此进行定义 TODO mkl transpose
// TAT_USE_SINGULAR_MATRIX svd出来的奇异值使用矩阵表示 TODO 没有设置此值时singular无法文本读入
// TAT_USE_SIMPLE_NAME 定义以使用原始字符串作为name
// TAT_USE_SIMPLE_NOSYMMETRY 定义以使用简单的Size作为无对称性的边
// TAT_USE_COPY_WITHOUT_WARNING 复制数据的时候不产生警告
// TAT_USE_VALID_DEFAULT_TENSOR 默认tensor初始化会产生一个合法的tensor, 默认不合法
// TAT_USE_EASY_CONVERSION tensor的各个接口可以自动转化类型 TODO 目前并不是所有接口都支持之
// TAT_USE_NO_TIMER 禁用对常见操作进行计时
// TAT_L3_CACHE, TAT_L2_CACHE, TAT_L1_CACHE 在转置中会使用
// TAT_USE_L3_CACHE 转置中默认不使用l3_cache, 设置以使用之

/**
 * \brief TAT is A Tensor library
 *
 * 张量含有两个模板参数, 分别是标量类型和对称性类型, 含有元信息和数据两部分. 元信息包括秩, 以及秩个边
 * 每个边含有一个Name信息以及形状信息, 对于无对称性的张量, 边的形状使用一个数字描述, 即此边的维度.
 * 对于其他类型的对称性, 边的形状为一个该类型对称性(应该是该对称性的量子数, 这里简称对称性)到数的映射,
 * 表示某量子数下的维度. 而张量数据部分为若干个秩维矩块, 对于无对称性张量, 仅有唯一一个矩块.
 */
namespace TAT {
   /**
    * \brief TAT的版本号
    */
   inline const char* version =
#ifdef TAT_VERSION
         TAT_VERSION
#else
         "unknown"
#endif
         ;

   /**
    * \brief 编译与license相关的信息
    */
   const char* information = "TAT"
#ifdef TAT_VERSION
                             " " TAT_VERSION
#endif
                             " ("
#ifdef TAT_BUILD_TYPE
                             "" TAT_BUILD_TYPE ", "
#endif
                             "" __DATE__ ", " __TIME__
#ifdef TAT_COMPILER_INFORMATION
                             ", " TAT_COMPILER_INFORMATION
#endif
                             ")\n"
                             "Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
                             "This is free software; see the source for copying conditions.  There is NO\n"
                             "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.";

   /**
    * \brief Debug模式中, 将在程序末尾打印一行友情提示, 过早的优化是万恶之源, 同时控制windows下终端的色彩模式
    */
   struct evil_t {
      evil_t() noexcept;
      ~evil_t();
   };
   inline const evil_t evil;

   // 目前只在timer中用到了
   inline void TAT_log(const char* message);

   // 下面三个函数常用来被各种细分的警告或错误使用，可以通过设置他们为下面三个中的一个来选择对错误的容忍程度
   inline void TAT_nothing(const char*) {}

   /**
    * \brief TAT使用的打印警告
    * \param message 待打印的内容
    */
   inline void TAT_warning(const char* message);

   /**
    * \brief TAT使用的抛出运行时异常
    * \param message 异常说明
    */
   inline void TAT_error(const char* message);

   constexpr auto TAT_warning_or_error_when_copy_data =
#ifdef TAT_USE_COPY_WITHOUT_WARNING
         TAT_nothing;
#else
         TAT_warning;
#endif
   constexpr auto TAT_warning_or_error_when_inplace_scalar = TAT_warning;
   constexpr auto TAT_warning_or_error_when_inplace_transform = TAT_warning;
   constexpr auto TAT_warning_or_error_when_multiple_name_missing = TAT_warning;
   constexpr auto TAT_warning_or_error_when_lapack_error = TAT_warning;

   // 张量转置中会使用这三个变量
   constexpr unsigned long l1_cache =
#ifdef TAT_L1_CACHE
         TAT_L1_CACHE
#else
         98304
#endif
         ;
   constexpr unsigned long l2_cache =
#ifdef TAT_L2_CACHE
         TAT_L2_CACHE
#else
         786432
#endif
         ;
   constexpr unsigned long l3_cache =
#ifdef TAT_L3_CACHE
         TAT_L3_CACHE
#else
         4718592
#endif
         ;
} // namespace TAT

// clang-format off
#include "tensor.hpp"
#include "implement.hpp"
#include "tools.hpp"
// clang-format on

#endif
