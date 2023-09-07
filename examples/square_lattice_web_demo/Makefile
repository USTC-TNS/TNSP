INCLUDE=-I ../../include
LAPACK=../../emscripten/liblapack.a ../../emscripten/libblas.a ../../emscripten/libf2c.a
OPTION=-O3 -gsource-map -s EMULATE_FUNCTION_POINTER_CASTS=1 -std=c++17
kernel.js: kernel.cpp
	em++ -s EXPORTED_FUNCTIONS='["_create_lattice","_update_lattice","_get_energy","_get_spin","_get_den"]' -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' $(OPTION) $(INCLUDE) $(LAPACK) -o $@ $<

clean:
	rm -f kernel.js kernel.wasm kernel.wasm.map
