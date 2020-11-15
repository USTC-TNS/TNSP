/**
 * \file timer.hpp
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
#ifndef TAT_TIMER_HPP
#define TAT_TIMER_HPP

#include <chrono>
#include <stack>

namespace TAT {
#ifndef TAT_USE_NO_TIMER
   using time_point = std::chrono::high_resolution_clock::time_point;
   using time_duration = std::chrono::high_resolution_clock::duration;
   using time_pair = std::tuple<time_point, time_duration>;

   inline auto get_current_time() {
      return std::chrono::high_resolution_clock::now();
   }
   inline auto count_to_second(const time_duration& count) {
      return std::chrono::duration<double>(count).count();
   }

   struct timer_stack_t {
      std::stack<time_pair, std::vector<time_pair>> stack;
      timer_stack_t() {
         stack.push({get_current_time(), time_duration::zero()});
      }
      ~timer_stack_t() {
         const auto& [start_point, children_time] = stack.top();
         auto program_total_time = get_current_time() - start_point;
         auto program_misc_time = program_total_time - children_time;
         TAT_log(("total : " + std::to_string(count_to_second(program_total_time)) + ", " + std::to_string(count_to_second(program_misc_time)))
                       .c_str());
      }
   };
   inline auto timer_stack = timer_stack_t();

   struct timer {
      std::string timer_name;
      time_duration timer_total_count;
      time_duration timer_self_count;

      timer(const char* name) : timer_name(name), timer_total_count(time_duration::zero()), timer_self_count(time_duration::zero()) {}

      ~timer() {
         if (timer_total_count.count() != 0) {
            auto self_count_in_second = std::chrono::duration<float>(timer_self_count).count();
            auto total_count_in_second = std::chrono::duration<float>(timer_total_count).count();
            TAT_log((timer_name + " : " + std::to_string(total_count_in_second) + ", " + std::to_string(self_count_in_second)).c_str());
         }
      }

      struct timer_guard {
         timer* owner;

         timer_guard(timer* owner) : owner(owner) {
            timer_stack.stack.push({get_current_time(), time_duration::zero()});
         }

         ~timer_guard() {
            const auto& [start_point, children_time] = timer_stack.stack.top();
            auto this_guard_time = get_current_time() - start_point;
            owner->timer_total_count += this_guard_time;
            owner->timer_self_count += this_guard_time - children_time;
            timer_stack.stack.pop();
            std::get<1>(timer_stack.stack.top()) += this_guard_time;
         }
      };

      auto operator()() {
         return timer_guard(this);
      }
   };
#else
   struct timer {
      struct timer_guard {};
      auto operator()() {
         return nullptr;
      }
   };
#endif

#define TAT_DEFINE_TIMER(x) inline timer x##_guard(#x);
   TAT_DEFINE_TIMER(contract)
   TAT_DEFINE_TIMER(contract_kernel)
   TAT_DEFINE_TIMER(svd)
   TAT_DEFINE_TIMER(svd_kernel)
   TAT_DEFINE_TIMER(qr)
   TAT_DEFINE_TIMER(qr_kernel)
   TAT_DEFINE_TIMER(scalar_outplace)
   TAT_DEFINE_TIMER(scalar_inplace)
   TAT_DEFINE_TIMER(transpose)
   TAT_DEFINE_TIMER(transpose_kernel)
   TAT_DEFINE_TIMER(transpose_kernel_core)
   TAT_DEFINE_TIMER(multiple)
   TAT_DEFINE_TIMER(conjugate)
   TAT_DEFINE_TIMER(trace)
   TAT_DEFINE_TIMER(slice)
   TAT_DEFINE_TIMER(expand)
#undef TAT_DEFINE_TIMER
} // namespace TAT
#endif
