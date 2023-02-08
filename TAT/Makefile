done: compile_information.o \
			const_integral.o \
			empty_list.o \
			hash_absorb.o \
			overloaded.o \
			scalar_traits.o \
			timer.o \
			type_alias.o \
			unordered_parameter.o \
			windows_console_guard.o

todo: log.o

%.o: %.c++
	g++ -std=c++20 -fmodules-ts -c $< -o $@
