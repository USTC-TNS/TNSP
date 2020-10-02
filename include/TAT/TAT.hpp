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
// TAT_USE_SIMPLE_NAME 定义以使用原始字符串作为name

namespace TAT {
   /**
    * \brief TAT的版本号
    */
   inline const char* version = "0.0.5";

   /**
    * \brief Debug模式中, 将在程序末尾打印一行友情提示, 过早的优化是万恶之源, 同时控制windows下终端的色彩模式
    */
   struct evil_t {
      evil_t() {
#ifdef _WIN32
         HANDLE output_handle = GetStdHandle(STD_OUTPUT_HANDLE);
         DWORD output_mode = 0;
         GetConsoleMode(output_handle, &output_mode);
         SetConsoleMode(output_handle, output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
         HANDLE error_handle = GetStdHandle(STD_ERROR_HANDLE);
         DWORD error_mode = 0;
         GetConsoleMode(error_handle, &error_mode);
         SetConsoleMode(error_handle, error_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
      }
      ~evil_t();
   };
   inline const evil_t evil;
} // namespace TAT

// clang-format off
#include "tensor.hpp"
#include "implement.hpp"
#include "tools.hpp"
// clang-format on

#endif
