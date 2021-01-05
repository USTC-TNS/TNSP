/**
 * \file basic.hpp
 *
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef SQUARE_BASIC_HPP
#define SQUARE_BASIC_HPP

#include <TAT/TAT.hpp>
#include <memory>
#include <random>
#include <stdexcept>

namespace square {
   auto clear_line = "\u001b[2K";

   class NotImplementedError : public std::logic_error {
      using std::logic_error::logic_error;
   };

   using Size = TAT::Size;
   using Name = TAT::DefaultName;
   template<typename T>
   using Tensor = TAT::Tensor<T>;
   template<typename T>
   using real = TAT::real_base_t<T>;
   template<typename T>
   using complex = std::complex<real<T>>;
   template<typename T>
   using real_complex = std::conditional_t<TAT::is_complex_v<T>, real<T>, complex<T>>;

   template<typename O, typename I>
   O scalar_to(I input) {
      if constexpr (TAT::is_complex_v<I> && TAT::is_real_v<O>) {
         return input.real();
      } else {
         return input;
      }
   }

   template<typename T>
   T conj(T input) {
      if constexpr (TAT::is_complex_v<T>) {
         return input.conj();
      } else {
         return input;
      }
   }

   template<typename T>
   struct Common {
      using C = complex<T>;

      static std::shared_ptr<const Tensor<T>> Sx() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            auto tensor = Tensor<T>({"I0", "O0"}, {2, 2});
            tensor.zero();
            tensor.at({{"I0", 0}, {"O0", 1}}) = +0.5;
            tensor.at({{"I0", 1}, {"O0", 0}}) = +0.5;
            result = std::make_shared<const Tensor<T>>(std::move(tensor));
         }
         return result;
      }

      static std::shared_ptr<const Tensor<C>> Sy() {
         static std::shared_ptr<const Tensor<C>> result = nullptr;
         if (!result) {
            auto tensor = Tensor<C>({"I0", "O0"}, {2, 2});
            tensor.zero();
            tensor.at({{"I0", 0}, {"O0", 1}}) = C{0, -0.5};
            tensor.at({{"I0", 1}, {"O0", 0}}) = C{0, +0.5};
            result = std::make_shared<const Tensor<C>>(std::move(tensor));
         }
         return result;
      }

      static std::shared_ptr<const Tensor<T>> Sz() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            auto tensor = Tensor<T>({"I0", "O0"}, {2, 2});
            tensor.zero();
            tensor.at({{"I0", 0}, {"O0", 0}}) = +0.5;
            tensor.at({{"I0", 1}, {"O0", 1}}) = -0.5;
            result = std::make_shared<const Tensor<T>>(std::move(tensor));
         }
         return result;
      }

      static std::shared_ptr<const Tensor<T>> SxSx() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            auto single = Sx();
            result = std::make_shared<const Tensor<T>>(single->edge_rename({{"I0", "I1"}, {"O0", "O1"}}).contract_all_edge(*single));
         }
         return result;
      }

      static std::shared_ptr<const Tensor<T>> SySy() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            auto single = Sy();
            result = std::make_shared<const Tensor<T>>(single->edge_rename({{"I0", "I1"}, {"O0", "O1"}}).contract_all_edge(*single).template to<T>());
         }
         return result;
      }

      static std::shared_ptr<const Tensor<T>> SzSz() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            auto single = Sz();
            result = std::make_shared<const Tensor<T>>(single->edge_rename({{"I0", "I1"}, {"O0", "O1"}}).contract_all_edge(*single));
         }
         return result;
      }

      static std::shared_ptr<const Tensor<T>> SS() {
         static std::shared_ptr<const Tensor<T>> result = nullptr;
         if (!result) {
            result = std::make_shared<const Tensor<T>>(*SxSx() + *SySy() + *SzSz());
         }
         return result;
      }
   };

   namespace random {
      inline auto engine = std::default_random_engine();

      /**
       * 获取随机数种子
       *
       * 保证各个mpi进程返回值相同, 如果root进程输入非0, 则使用root进程的输入作为种子, 则生成一个,
       */
      std::uint32_t get_seed(std::uint32_t seed = 0) {
         if (TAT::mpi.rank == 0) {
            if (seed == 0) {
               seed = std::random_device()();
            }
         }
         MPI_Bcast(&seed, /*count*/ 1, MPI_UINT32_T, /*root*/ 0, MPI_COMM_WORLD);
         return seed;
      }

      /**
       * 设置随机数种子
       *
       * 如果各个进程输入值都非0, 则格子设置随机数种子, 所以如果输入值为相同的非零值, 则会设置相同的随机数种子.
       * 如果各个进程输入值均为0, 则生成一个相同的真随机种子.
       * 如果有些进程为0, 有些非0, 则此函数会卡住
       */
      inline void seed(std::uint32_t seed) {
         if (seed == 0) {
            seed = get_seed();
         }
         engine.seed(seed);
         TAT::mpi.out_one() << "Set the same random seed to " << seed << "\n";
      }

      inline void split_seed() {
         auto new_seed = std::uniform_int_distribution<std::uint32_t>(1)(engine);
         TAT::mpi.out_one() << "Split random seed\n";
         engine.seed(new_seed + TAT::mpi.rank);
      }
      inline void merge_seed() {
         auto new_seed = std::uniform_int_distribution<std::uint32_t>(1)(engine);
         TAT::mpi.out_one() << "Merge random seed\n";
         engine.seed(get_seed(new_seed));
      }

      template<typename T>
      auto normal(T mean, T stddev) {
         if constexpr (TAT::is_complex_v<T>) {
            return [distribution_real = std::normal_distribution<real<T>>(mean.real(), stddev.real()),
                    distribution_imag = std::normal_distribution<real<T>>(mean.imag(), stddev.imag())]() mutable -> T {
               return {distribution_real(engine), distribution_imag(engine)};
            };
         } else {
            return [distribution = std::normal_distribution<T>(mean, stddev)]() mutable { return distribution(engine); };
         }
      }
      template<typename T>
      auto uniform(T min, T max) {
         if constexpr (std::is_integral_v<T>) {
            return [distribution = std::uniform_int_distribution<T>(min, max)]() mutable { return distribution(engine); };
         } else if constexpr (TAT::is_complex_v<T>) {
            return [distribution_real = std::uniform_real_distribution<real<T>>(min.real(), max.real()),
                    distribution_imag = std::uniform_real_distribution<real<T>>(min.imag(), max.imag())]() mutable -> T {
               return {distribution_real(engine), distribution_imag(engine)};
            };
         } else {
            return [distribution = std::uniform_real_distribution<T>(min, max)]() mutable { return distribution(engine); };
         }
      }
   } // namespace random

   template<typename T>
   struct ExactLattice;
   template<typename T>
   struct SimpleUpdateLattice;
   template<typename T>
   struct SamplingGradientLattice;

   template<typename T>
   std::istream&& operator>(std::istream&& in, T& v) {
      in > v;
      return std::move(in);
   }
   template<typename T>
   std::ostream&& operator<(std::ostream&& out, const T& v) {
      out < v;
      return std::move(out);
   }

   template<typename T>
   MPI_Datatype get_mpi_type();

   template<>
   MPI_Datatype get_mpi_type<float>() {
      return MPI_FLOAT;
   }
   template<>
   MPI_Datatype get_mpi_type<double>() {
      return MPI_DOUBLE;
   }
   template<>
   MPI_Datatype get_mpi_type<complex<float>>() {
      return MPI_C_FLOAT_COMPLEX;
   }
   template<>
   MPI_Datatype get_mpi_type<complex<double>>() {
      return MPI_C_DOUBLE_COMPLEX;
   }
} // namespace square

#endif
