/**
 * \file mpi.hpp
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
#ifndef TAT_MPI_HPP
#define TAT_MPI_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "io.hpp"
#include "tensor.hpp"

#ifdef TAT_USE_MPI
#include <mpi.h>
#endif

namespace TAT {
#ifdef TAT_USE_MPI
   struct mpi_t {
      int size;
      int rank;
      static bool initialized() noexcept {
         int result;
         MPI_Initialized(&result);
         return result;
      }
      static bool finalized() noexcept {
         int result;
         MPI_Finalized(&result);
         return result;
      }
      mpi_t() noexcept : size(1), rank(0) {
         if (!initialized()) {
            MPI_Init(nullptr, nullptr);
         }
         MPI_Comm_size(MPI_COMM_WORLD, &size);
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
      ~mpi_t() {
         if (!finalized()) {
            MPI_Finalize();
         }
      }
   };
   inline mpi_t mpi;

   /**
    * \brief source调用此函数, 向destination发送一个张量
    */
   template<class ScalarType, class Symmetry>
   void send(const Tensor<ScalarType, Symmetry>& tensor, const int destination) {
      auto data = tensor.dump(); // TODO: 也许可以不需复制, 但这个在mpi框架内可能不是很方便
      MPI_Send(data.data(), data.length(), MPI_BYTE, destination, 0, MPI_COMM_WORLD);
   }

   // TODO: 异步的处理, 这个优先级很低, 也许以后将和gpu中做svd, gemm一起做成异步
   /**
    * \brief destination调用此函数, 从source接受一个张量
    */
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> receive(const int source) {
      auto status = MPI_Status();
      MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
      int length;
      MPI_Get_count(&status, MPI_BYTE, &length);
      auto data = std::string(length, '\0'); // 这里不需初始化, 但考虑到load和dump本身效率也不高, 无所谓了
      MPI_Recv(data.data(), length, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      auto result = Tensor<ScalarType, Symmetry>().load(data);
      return result;
   }

   /**
    * \brief destination调用此函数, 从source接受一个张量
    * 第一个参数并不使用, 仅仅是为了确定张量类型
    */
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> receive(const Tensor<ScalarType, Symmetry>&, const int source) {
      return receive<ScalarType, Symmetry>(source);
   }

   /**
    * 像简单类型一样使用mpi但send和receive, 调用后, 一个destination返回source调用时输入tensor, 其他进程返回空张量
    */
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> send_receive(const Tensor<ScalarType, Symmetry>& tensor, const int source, const int destination) {
      if (mpi.rank == source) {
         send(tensor, destination);
      }
      if (mpi.rank == destination) {
         return receive(tensor, source);
      }
      return Tensor<ScalarType, Symmetry>();
   }

   /**
    * 从root进程分发张量, 使用简单的树形分发, 必须所有进程一起调用这个函数
    */
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> broadcast(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
      if (mpi.size == 1) {
         return tensor.copy(); // rvalue
      }
      if (0 > root || root >= mpi.size) {
         TAT_error("Invalid root rank when mpi broadcast a tensor");
      }
      const auto this_fake_rank = (mpi.size + mpi.rank - root) % mpi.size;
      Tensor<ScalarType, Symmetry> result;
      // get from father
      if (this_fake_rank != 0) {
         const auto father_fake_rank = (this_fake_rank - 1) / 2;
         const auto father_real_rank = (father_fake_rank + root) % mpi.size;
         result = receive(result, father_real_rank);
      } else {
         // 自己就是root的话, 会复制一次张量
         result = tensor.copy();
      }
      // send to son
      const auto left_son_fake_rank = this_fake_rank * 2 + 1;
      const auto right_son_fake_rank = this_fake_rank * 2 + 2;
      if (left_son_fake_rank < mpi.size) {
         const auto left_son_real_rank = (left_son_fake_rank + root) % mpi.size;
         send(result, left_son_real_rank);
      }
      if (right_son_fake_rank < mpi.size) {
         const auto right_son_real_rank = (right_son_fake_rank + root) % mpi.size;
         send(result, right_son_real_rank);
      }
      return result;
   }

   /**
    * 向root进程reduce张量, 使用简单的树形reduce, 必须所有进程一起调用这个函数, 最后root进程返回全部reduce的结果, 其他进程为中间结果一般无意义
    */
   template<class ScalarType, class Symmetry, class Func>
   Tensor<ScalarType, Symmetry> reduce(const Tensor<ScalarType, Symmetry>& tensor, const int root, Func&& function) {
      if (mpi.size == 1) {
         return tensor.copy(); // rvalue
      }
      if (0 > root || root >= mpi.size) {
         TAT_error("Invalid root rank when mpi reduce a tensor");
      }
      const auto this_fake_rank = (mpi.size + mpi.rank - root) % mpi.size;
      Tensor<ScalarType, Symmetry> result;
      // get from son
      const auto left_son_fake_rank = this_fake_rank * 2 + 1;
      const auto right_son_fake_rank = this_fake_rank * 2 + 2;
      if (left_son_fake_rank < mpi.size) {
         const auto left_son_real_rank = (left_son_fake_rank + root) % mpi.size;
         result = function(tensor, receive(result, left_son_real_rank));
      }
      if (right_son_fake_rank < mpi.size) {
         const auto right_son_real_rank = (right_son_fake_rank + root) % mpi.size;
         // 如果左儿子不存在, 那么右儿子一定不存在, 所以不必判断result是否有效
         result = function(result, receive(result, right_son_real_rank));
      }
      // pass to father
      if (this_fake_rank != 0) {
         const auto father_fake_rank = (this_fake_rank - 1) / 2;
         const auto father_real_rank = (father_fake_rank + root) % mpi.size;
         if (left_son_fake_rank < mpi.size) {
            send(result, father_real_rank);
         } else {
            send(tensor, father_real_rank);
         }
      }
      return result;
      // 子叶为空tensor, 每个非子叶节点为reduce了所有的后代的结果
   }

   /**
    * 对各个进程但张量通过求和进行reduce
    */
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> summary(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
      return reduce(tensor, root, [](const auto& tensor_1, const auto& tensor_2) { return tensor_1 + tensor_2; });
   }

   /**
    * mpi进程间同步
    */
   inline void barrier() {
      MPI_Barrier(MPI_COMM_WORLD);
   }

   struct mpi_output_stream {
      std::ostream* out;
      int rank;
      mpi_output_stream(std::ostream* out, int rank = 0) : out(out), rank(rank) {}

      template<class Type>
      mpi_output_stream& operator<<(const Type& value) & {
         if (mpi.rank == rank) {
            *out << value;
         }
         return *this;
      }

      template<class Type>
      mpi_output_stream&& operator<<(const Type& value) && {
         if (mpi.rank == rank) {
            *out << value;
         }
         return std::move(*this);
      }
   };

   inline auto mpi_out(int rank = 0) {
      return mpi_output_stream(&std::cout, rank);
   }
   inline auto mpi_log(int rank = 0) {
      return mpi_output_stream(&std::clog, rank);
   }
   inline auto mpi_err(int rank = 0) {
      return mpi_output_stream(&std::cerr, rank);
   }
#endif

   inline evil_t::evil_t() noexcept {
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
   inline evil_t::~evil_t() {
#ifndef NDEBUG
      try {
#ifdef TAT_USE_MPI
         mpi_log()
#else
         std::clog
#endif
               << console_blue << "\n\nPremature optimization is the root of all evil!\n"
               << console_origin << "                                       --- Donald Knuth\n\n\n";
      } catch (const std::exception&) {
      }
#endif
   }

   inline void TAT_warning(const char* message) {
      std::cerr << console_red;
#ifdef TAT_USE_MPI
      if (mpi.size != 1) {
         std::cerr << "[rank " << mpi.rank << "] ";
      }
#endif
      std::cerr << message << console_origin << std::endl;
   }

   inline void TAT_error(const char* message) {
      throw std::runtime_error(message);
   }
} // namespace TAT
#endif
