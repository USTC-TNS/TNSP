done: compile_information.o \
			const_integral.o \
			empty_list.o \
			hash_absorb.o \
			overloaded.o \
			scalar_traits.o \
			timer.o \
			type_alias.o \
			unordered_parameter.o \
			windows_console_guard.o \
			shared_ptr.o \
			multidimension_span.o

todo: log_and_exception.o

%.o: %.c++
	g++ -std=c++20 -fmodules-ts -c $< -o $@

std.o: std.c++ sys

SYSROOT := gcm.cache/usr/include/c++/12.2.1

${SYSROOT}/%.gcm:
	g++ -std=c++20 -fmodules-ts -x c++-system-header -c $*

sys: \
${SYSROOT}/algorithm.gcm \
${SYSROOT}/array.gcm \
${SYSROOT}/chrono.gcm \
${SYSROOT}/cmath.gcm \
${SYSROOT}/complex.gcm \
${SYSROOT}/cstddef.gcm \
${SYSROOT}/cstdint.gcm \
${SYSROOT}/forward_list.gcm \
${SYSROOT}/iostream.gcm \
${SYSROOT}/limits.gcm \
${SYSROOT}/list.gcm \
${SYSROOT}/map.gcm \
${SYSROOT}/memory.gcm \
${SYSROOT}/numeric.gcm \
${SYSROOT}/optional.gcm \
${SYSROOT}/set.gcm \
${SYSROOT}/sstream.gcm \
${SYSROOT}/stack.gcm \
${SYSROOT}/stdexcept.gcm \
${SYSROOT}/string.gcm \
${SYSROOT}/string_view.gcm \
${SYSROOT}/tuple.gcm \
${SYSROOT}/type_traits.gcm \
${SYSROOT}/unordered_map.gcm \
${SYSROOT}/unordered_set.gcm \
${SYSROOT}/utility.gcm \
${SYSROOT}/variant.gcm \
${SYSROOT}/vector.gcm \

clean:
	rm -rf gcm.cache *.o
