# Welcome to the TAT Documents

[![version](https://img.shields.io/github/v/tag/hzhangxyz/TAT?style=flat-square)](https://github.com/hzhangxyz/TAT/tags)
[![license](https://img.shields.io/github/license/hzhangxyz/TAT?style=flat-square)](https://github.com/hzhangxyz/TAT/blob/TAT/LICENSE.md)
[![build](https://img.shields.io/github/workflow/status/hzhangxyz/TAT/check?style=flat-square)](https://github.com/hzhangxyz/TAT/actions?query=workflow%3Acheck)
[![document](https://img.shields.io/github/workflow/status/hzhangxyz/TAT/doxygen?color=%237f7fff&label=doc&style=flat-square)](https://hzhangxyz.github.io/TAT/index.html)

[TAT](https://github.com/hzhangxyz/TAT) is a header-only c++ tensor library with support for Abelian [symmetry](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.050301) for tensor network

The name "TAT" is a recursive acronym for "TAT is A Tensor library!", and it should be all uppercase

## Prerequisites
- c++ compiler with c++17 support(such as gcc7+, clang5+, msvc19.14+)
- lapack/blas or mkl
- [pybind11](https://github.com/pybind/pybind11)(optional for python binding)
- mpi(optional for parallel computing)

## Usage
Just include the file `include/TAT/TAT.hpp` and link lapack/blas or mkl at link time

For good practice, pass argument `-I$path_to_TAT_root/include` to compiler and use `#include <TAT/TAT.hpp>` in your source file

For mpi support, you need to define macro `TAT_USE_MPI` and use mpi compiler such as `mpic++`(recommend) or pass correct flag to normal compiler(for expert)

Please check comment in file `TAT.hpp` for some other macro options

Please notice that this library need proper compiler optimization option(`-O2`, `-O3`, `-Ofast`) for good performace

You can also use TAT as a cmake subdirectory, just use `add_subdirectory(path_to_TAT_root)` and `target_link_libraries(your_target TAT)` in your `CMakeLists.txt`

## Python binding
Python binding is configured in `CMakeLists.txt`, use cmake and build target `PyTAT`

For other customed python module name, define `TAT_PYTHON_MODULE` as cmake variable

If you are familiar with [pybind11](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually), you can compile [`python/TAT.cpp`](/python/TAT.cpp) directly with correct flag

Please notice that for some old mpi version(for example, openmpi 2.1.1 in ubuntu 18.04 LTS), you need to load mpi dynamic shared library manually before `import TAT`, The way to load it manually is `import ctypes` and `ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)`

## Use with [emscripten](https://emscripten.org/)
If you want to run a program using TAT in browser, which is very useful for demonstration

You can simply compile TAT with `em++`(no mpi support, no doubt), and link `liblapack.a`, `libblas.a`, `libf2c.a` compiled from [clapack-3.2.1](https://www.netlib.org/clapack/)

You can download them from [here](https://github.com/hzhangxyz/TAT/releases/tag/v0.0.6) or compile by yourself

If you are using cmake, you need to put these three files into directory `emscripten`, then run `emcmake cmake $path_to_TAT_root` which will configure it automatically

## Installation (for Arch Linux users only)
Download PKGBUILD into any empty directory and run `makepkg` to build your package

Then use `pacman -U TAT-xxx-x86_64.pkg.tar.zst` to install

## Links
- [a tensor network library](https://github.com/crimestop/net)
- [gitee mirror of TAT](https://gitee.com/hzhangxyz/TAT)
