export module TAT.timer;

import <chrono>;
import <stack>;
import <string>;
import <utility>;
import <vector>;

import TAT.compile_information;
import TAT.log_and_exception;

namespace TAT {
   using time_point = std::chrono::high_resolution_clock::time_point;
   using time_duration = std::chrono::high_resolution_clock::duration;
   using time_pair = std::pair<time_point, time_duration>;

   struct timer_stack_t {
      std::stack<time_pair, std::vector<time_pair>> stack;
      timer_stack_t();
      ~timer_stack_t();
   };
   /**
    * Timer type
    *
    * Use single timer stack to count time cost different function concerned, including itself and total.
    *
    * To use it, define a global timer `timer xxx_guard("xxx")`, and call `auto timer_guard = xxx_guard();`
    * at the begin of the function
    */
   export struct timer {
      friend struct timer_guard_t;
    private:
      std::string timer_name;
      time_duration timer_total_count;
      time_duration timer_self_count;
    public:
      timer(const auto& name);
      ~timer();

      /**
       * Start to count time for the timer in current scope
       */
      auto operator()();
   };
   struct timer_guard_t {
    private:
      timer* owner;
    public:
      timer_guard_t(timer* owner);
      ~timer_guard_t();
   };

   auto get_current_time() {
      return std::chrono::high_resolution_clock::now();
   }
   auto duration_to_second(const time_duration& count) {
      return std::chrono::duration<double>(count).count();
   }

   timer_stack_t::timer_stack_t() {
      stack.push({get_current_time(), time_duration::zero()});
   }
   timer_stack_t::~timer_stack_t() {
      const auto& [start_point, children_time] = stack.top(); // there should be only one item in the stack;
      auto program_total_time = get_current_time() - start_point;
      auto program_misc_time = program_total_time - children_time;
      log("total : " + std::to_string(duration_to_second(program_total_time)) + ", " + std::to_string(duration_to_second(program_misc_time)));
   }

   timer::timer(const auto& name) : timer_name(name), timer_total_count(time_duration::zero()), timer_self_count(time_duration::zero()) {}
   timer::~timer() {
      if (timer_total_count.count() != 0) {
         auto self_count_in_second = std::chrono::duration<float>(timer_self_count).count();
         auto total_count_in_second = std::chrono::duration<float>(timer_total_count).count();
         log(timer_name + " : " + std::to_string(total_count_in_second) + ", " + std::to_string(self_count_in_second));
      }
   }
   auto timer::operator()() {
      if constexpr (use_timer) {
         return timer_guard_t(this);
      } else {
         return 0;
      }
   }

   inline auto timer_stack = timer_stack_t();

   timer_guard_t::timer_guard_t(timer* owner) : owner(owner) {
      timer_stack.stack.push({get_current_time(), time_duration::zero()});
   }
   timer_guard_t::~timer_guard_t() {
      const auto& [start_point, children_time] = timer_stack.stack.top();
      auto this_guard_time = get_current_time() - start_point;
      owner->timer_total_count += this_guard_time;
      owner->timer_self_count += this_guard_time - children_time;
      timer_stack.stack.pop();
      timer_stack.stack.top().second += this_guard_time;
   }

} // namespace TAT
