#!/usr/bin/env bash

em++ -I ../../include/ ../../emscripten/liblapack.a ../../emscripten/libblas.a ../../emscripten/libf2c.a -std=c++17 kernel.cpp -s EXPORTED_FUNCTIONS='["_create_lattice","_update_lattice","_get_energy","_get_spin","_get_den"]' -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' -O3 -o kernel.js
