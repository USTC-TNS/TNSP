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

#include "io.hpp"
#include "tensor.hpp"

#ifdef TAT_USE_MPI
#include <mpi.h>
#endif

namespace TAT {
#ifdef TAT_USE_MPI
   namespace mpi {
      struct mpi_t {
         int size;
         int rank;
         static bool initialized() {
            int result;
            MPI_Initialized(&result);
            return result;
         }
         static bool finalized() {
            int result;
            MPI_Finalized(&result);
            return result;
         }
         mpi_t() {
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

      template<class ScalarType, class Symmetry>
      void send(const Tensor<ScalarType, Symmetry>& tensor, const int destination) {
         auto out = std::stringstream();
         out < tensor;
         auto data = out.str(); // TODO: 不需复制
         MPI_Send(data.data(), data.length(), MPI_BYTE, destination, 0, MPI_COMM_WORLD);
      }

      // TODO: 异步
      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> receive(const Tensor<ScalarType, Symmetry>&, const int source) {
         // Tensor只是为了确定template参数
         auto status = MPI_Status();
         MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
         int length;
         MPI_Get_count(&status, MPI_BYTE, &length);
         auto data = std::string(length, '\0'); // 不需初始化
         MPI_Recv(data.data(), length, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         auto in = std::stringstream(data); // 不需复制
         auto result = Tensor<ScalarType, Symmetry>();
         in > result;
         return result;
      }

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

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> broadcast(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
         if (mpi.size == 1) {
            return tensor.copy(); // rvalue
         }
         if (0 > root || root >= mpi.size) {
            throw TAT_error("Invalid root rank when mpi broadcast a tensor");
         }
         const auto this_fake_rank = (mpi.size + mpi.rank - root) % mpi.size;
         Tensor<ScalarType, Symmetry> result;
         // get from father
         if (this_fake_rank != 0) {
            const auto father_fake_rank = (this_fake_rank - 1) / 2;
            const auto father_real_rank = (father_fake_rank + root) % mpi.size;
            result = receive(result, father_real_rank);
         } else {
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

      template<class ScalarType, class Symmetry, class Func>
      Tensor<ScalarType, Symmetry> reduce(const Tensor<ScalarType, Symmetry>& tensor, const int root, Func&& function) {
         if (mpi.size == 1) {
            return tensor.copy(); // rvalue
         }
         if (0 > root || root >= mpi.size) {
            throw TAT_error("Invalid root rank when mpi reduce a tensor");
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

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> summary(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
         return reduce(tensor, root, [](const auto& tensor_1, const auto& tensor_2) { return tensor_1 + tensor_2; });
      }

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
   } // namespace mpi
#endif

   inline evil_t::~evil_t() {
#ifndef NDEBUG
      try {
#ifdef TAT_USE_MPI
         mpi::mpi_log()
#else
         std::clog
#endif
               << console_blue << "\n\nPremature optimization is the root of all evil!\n"
               << console_origin << "                                       --- Donald Knuth\n\n\n";
      } catch (const std::exception&) {
      }
#endif
   }

   inline void warning_or_error([[maybe_unused]] const char* message) {
#ifndef NDEBUG
      std::cerr << console_red
#ifdef TAT_USE_MPI
                << "rank " << mpi::mpi.rank << " : "
#endif
                << message << console_origin << std::endl;
#endif
   }
} // namespace TAT
#endif
