module;

export module TAT.overloaded;

namespace TAT {
   export template<typename... Fs>
   struct overloaded : Fs... {
      using Fs::operator()...;
   };
   export template<typename... Fs>
   overloaded(Fs...) -> overloaded<Fs...>;
} // namespace TAT
