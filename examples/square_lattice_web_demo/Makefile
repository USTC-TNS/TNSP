PYODIDE_VERSION=0.25.0a2
INCLUDE=-I ../../include
OPTION=-O3 -gsource-map -s EMULATE_FUNCTION_POINTER_CASTS=1 -std=c++17 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=loadWASM -s ALLOW_MEMORY_GROWTH=1

default: kernel.js

xbuildenv-${PYODIDE_VERSION}.tar.bz2:
	wget https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/$@

libopenblas.a: xbuildenv-${PYODIDE_VERSION}.tar.bz2
	tar xf $< xbuildenv/pyodide-root/packages/.libs/lib/$@ --strip-components=5

kernel.js: kernel.cpp libopenblas.a
	em++ $(OPTION) $(INCLUDE) -o $@ $^

clean:
	rm -f kernel.js kernel.wasm kernel.wasm.map

clean-all:
	rm -f kernel.js kernel.wasm kernel.wasm.map libopenblas.a *.bz2
