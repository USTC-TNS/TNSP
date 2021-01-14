/**
 * \file TAT.hpp
 *
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#define NOMINMAX
#include <windows.h>
#endif

// 开关说明
// TAT_USE_MPI 定义以开启MPI支持, cmake可对此进行定义
// TAT_USE_MKL_TRANSPOSE 定义以使用mkl加速转置, cmake可对此进行定义 TODO 进一步优化
// TAT_USE_MKL_GEMM_BATCH 定义以使用mkl的?gemm_batch, cmake可对此进行定义
// TAT_USE_SINGULAR_MATRIX svd出来的奇异值使用矩阵表示
// TAT_USE_SIMPLE_NAME 定义以使用原始字符串作为name
// TAT_USE_SIMPLE_NOSYMMETRY 定义以使用简单的Size作为无对称性的边
// TAT_USE_VALID_DEFAULT_TENSOR 默认tensor初始化会产生一个合法的tensor, 默认不合法
// TAT_USE_TIMER 对常见操作进行计时
// TAT_ERROR_BITS 将各类警告转换为异常
// TAT_NOTHING_BITS 将各类警告转换为静默
// TAT_L3_CACHE, TAT_L2_CACHE, TAT_L1_CACHE 在转置中会使用
// TAT_USE_L3_CACHE 转置中默认不使用l3_cache, 设置以使用之

/**
 * TAT is A Tensor library
 */
namespace TAT {
   /**
    * \defgroup Miscellaneous
    * @{
    */

   /**
    * TAT的版本号
    */
   inline const char* version =
#ifdef TAT_VERSION
         TAT_VERSION
#else
         "0.1.2"
#endif
         ;

   /**
    * 编译与license的相关信息
    */
   inline const char* information = "TAT"
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
                                    "Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
                                    "This is free software; see the source for copying conditions.  There is NO\n"
                                    "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.";

   /// \private
   struct evil_t {
      evil_t() noexcept;
      ~evil_t();
   };
   /**
    * Debug模式中, 将在程序末尾打印一行友情提示, 过早的优化是万恶之源, 同时此对象也控制windows下终端的色彩模式
    */
   inline const evil_t evil;

   /**
    * TAT使用的日志打印
    */
   inline void TAT_log(const char* message);

   /**
    * 什么事情也不做
    *
    * 接口和`TAT_warning`, `TAT_error`一致, 可供各种细分的警告或错误使用, 通过设置他们为这三个中的一个的指针来选择对错误的容忍程度
    *
    * \see TAT_warning, TAT_error
    */
   inline void TAT_nothing(const char*) {}

   /**
    * TAT使用的打印警告
    *
    * \param message 待打印的内容
    *
    * \see TAT_nothing, TAT_error
    */
   inline void TAT_warning(const char* message);

   /**
    * TAT使用的抛出运行时异常
    *
    * \param message 异常说明
    *
    * \see TAT_nothing, TAT_warning
    */
   inline void TAT_error(const char* message);

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS

#ifdef TAT_USE_NO_WARNING
// TODO delete this deprecated macro
#pragma message("TAT_USE_NO_WARNING is deprecated, define TAT_NOTHING_BITS=7 instead")
#define TAT_NOTHING_BITS 7
#endif

#ifndef TAT_ERROR_BITS
#define TAT_ERROR_BITS 0
#endif
#ifndef TAT_NOTHING_BITS
#define TAT_NOTHING_BITS 0
#endif

   constexpr auto TAT_warning_or_error_when_lapack_error = TAT_ERROR_BITS & 1 ? TAT_error : TAT_NOTHING_BITS & 1 ? TAT_nothing : TAT_warning;
   constexpr auto TAT_warning_or_error_when_name_missing = TAT_ERROR_BITS & 2 ? TAT_error : TAT_NOTHING_BITS & 2 ? TAT_nothing : TAT_warning;
   constexpr auto TAT_warning_or_error_when_copy_shared = TAT_ERROR_BITS & 4 ? TAT_error : TAT_NOTHING_BITS & 4 ? TAT_nothing : TAT_warning;
#endif

   /**
    * 供转置中使用的l1 cache大小, 可由宏`TAT_L1_CACHE`设置
    */
   constexpr unsigned long l1_cache =
#ifdef TAT_L1_CACHE
         TAT_L1_CACHE
#else
         98304
#endif
         ;
   /**
    * 供转置中使用的l2 cache大小, 可由宏`TAT_L2_CACHE`设置
    */
   constexpr unsigned long l2_cache =
#ifdef TAT_L2_CACHE
         TAT_L2_CACHE
#else
         786432
#endif
         ;
   /**
    * 供转置中使用的l3 cache大小, 可由宏`TAT_L3_CACHE`设置
    */
   constexpr unsigned long l3_cache =
#ifdef TAT_L3_CACHE
         TAT_L3_CACHE
#else
         4718592
#endif
         ;
   /**@}*/
} // namespace TAT

// clang-format off
#include "tensor.hpp"
#include "implement.hpp"
// clang-format on

#endif
