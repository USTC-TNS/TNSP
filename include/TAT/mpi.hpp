/**
 * \file mpi.hpp
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

      // TODO 可以优化
      template<class ScalarType, class Symmetry>
      void send(const Tensor<ScalarType, Symmetry>& tensor, const int source, const int destination) {
         auto out = ::std::stringstream();
         out <= tensor;
         auto data = out.str(); // 不需复制
         MPI_Send(data.data(), data.length(), MPI_BYTE, destination, 0, MPI_COMM_WORLD);
      }

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> receive(const Tensor<ScalarType, Symmetry>&, const int source, const int destination) {
         auto status = MPI_Status();
         MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
         int length;
         MPI_Get_count(&status, MPI_BYTE, &length);
         auto data = ::std::string(length, '\0'); // 不需初始化
         MPI_Recv(data.data(), length, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         auto in = ::std::stringstream(data); // 不需复制
         auto result = Tensor<ScalarType, Symmetry>();
         in >= result;
         // TODO STATUS?
         return result;
      }

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> send_receive(const Tensor<ScalarType, Symmetry>& tensor, const int source, const int destination) {
         if (mpi.rank == destination) {
            return receive(tensor, source, destination);
         }
         if (mpi.rank == source) {
            send(tensor, source, destination);
         }
         return Tensor<ScalarType, Symmetry>();
      }

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> broadcast(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
         if (mpi.size == 1) {
            return tensor.copy();
         }
         int length;
         auto data = ::std::string();
         // ::std::cout << mpi.rank << "rank\n";
         if (mpi.rank == root) {
            auto out = ::std::stringstream();
            out <= tensor;
            data = out.str();
            length = data.length();
         }
         // ::std::cout << mpi.rank << " " << length << "\n";
         MPI_Bcast(&length, 1, MPI_INT, root, MPI_COMM_WORLD);
         // ::std::cout << mpi.rank << " " << length << "\n";
         data.resize(length);
         MPI_Bcast(data.data(), length, MPI_BYTE, root, MPI_COMM_WORLD);
         auto in = ::std::stringstream(data);
         auto result = Tensor<ScalarType, Symmetry>();
         in >= result;
         return result;
      }

      template<class ScalarType, class Symmetry, class Func>
      Tensor<ScalarType, Symmetry> reduce(const Tensor<ScalarType, Symmetry>& tensor, const int root, Func&& function) {
         if (mpi.size == 1) {
            return tensor.copy(); // rvalue
         }
         int this_rank = (mpi.size + mpi.rank - root) % mpi.size;
         Tensor<ScalarType, Symmetry> result;
         // get from son
         int this_left_son_rank = this_rank * 2 + 1;
         int this_right_son_rank = this_rank * 2 + 2;
         if (this_left_son_rank < mpi.size) {
            int left_son_rank = (this_left_son_rank + root) % mpi.size;
            // ::std::cout << "receiving from " << left_son_rank << " to " << mpi.rank << "\n";
            result = function(tensor, receive(tensor, left_son_rank, mpi.rank));
            // ::std::cout << "received from " << left_son_rank << " to " << mpi.rank << "\n";
         }
         if (this_right_son_rank < mpi.size) {
            int right_son_rank = (this_right_son_rank + root) % mpi.size;
            // ::std::cout << "receiving from " << right_son_rank << " to " << mpi.rank << "\n";
            result = function(result, receive(tensor, right_son_rank, mpi.rank));
            // ::std::cout << "received from " << right_son_rank << " to " << mpi.rank << "\n";
         }
         // pass to father
         int father_rank = ((this_rank - 1) / 2 + mpi.size) % mpi.size;
         if (this_rank != 0) {
            // ::std::cout << "sending from " << mpi.rank << " to " << father_rank << "\n";
            if (this_right_son_rank > mpi.size) {
               send(tensor, mpi.rank, father_rank);
            } else {
               send(result, mpi.rank, father_rank);
            }
            // ::std::cout << "sent from " << mpi.rank << " to " << father_rank << "\n";
         }
         return result;
      }

      template<class ScalarType, class Symmetry>
      Tensor<ScalarType, Symmetry> summary(const Tensor<ScalarType, Symmetry>& tensor, const int root) {
         return reduce(tensor, root, [](const auto& tensor_1, const auto& tensor_2) { return tensor_1 + tensor_2; });
      }

      // TODO: scatter and gather not impl yet

      inline void barrier() {
         MPI_Barrier(MPI_COMM_WORLD);
      }

      struct mpi_output_stream {
         ::std::ostream* out;
         int rank;
         mpi_output_stream(::std::ostream* out, int rank = 0) : out(out), rank(rank) {}

         template<class Type>
         mpi_output_stream& operator<<(const Type& value) {
            if (mpi.rank == rank) {
               *out << value;
            }
            return *this;
         }
      };

      inline auto root_out = mpi_output_stream(&::std::cout, 0);
      inline auto root_log = mpi_output_stream(&::std::clog, 0);
      inline auto root_err = mpi_output_stream(&::std::cerr, 0);
   } // namespace mpi
#endif

   inline Evil::~Evil() {
#ifndef NDEBUG
      try {
#ifdef TAT_USE_MPI
         mpi::root_log
#else
         ::std::clog
#endif
               << console_blue << "\n\nPremature optimization is the root of all evil!\n"
               << console_origin << "                                       --- Donald Knuth\n\n\n";
      } catch (const ::std::exception&) {
      }
#endif
   }

   inline void warning_or_error([[maybe_unused]] const ::std::string& message) {
#ifndef NDEBUG
      ::std::cerr << console_red
#ifdef TAT_USE_MPI
                  << "rank " << mpi::mpi.rank << " : "
#endif
                  << message << console_origin << ::std::endl;
#endif
   }
} // namespace TAT
#endif
