# [TAT](https://github.com/hzhangxyz/TAT) &middot; [![version](https://img.shields.io/github/v/tag/hzhangxyz/TAT)](https://github.com/hzhangxyz/TAT/tags) [![license](https://img.shields.io/github/license/hzhangxyz/TAT)](/LICENSE) [![check](https://github.com/hzhangxyz/TAT/workflows/check/badge.svg)](https://github.com/hzhangxyz/TAT/actions?query=workflow%3Acheck)

TAT is a header-only c++ tensor library with support for Abelian [symmetry](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.050301) for tensor network

The name "TAT" is a recursive acronym for "TAT is A Tensor library!", and it should be all uppercase.

## Prerequisites
- c++ compiler with c++17 support(such as gcc7+)
- blas/lapack or mkl
- [pybind11](https://github.com/pybind/pybind11)(optional for python binding)
- mpi(optional for parallel computing)

## Usage
Just include the file [`include/TAT/TAT.hpp`](/include/TAT/TAT.hpp) and link blas/lapack or mkl in link time

For good practice, pass `-I$path_to_repo_root/include` to compiler and use `#include <TAT/TAT.hpp>` in your source file

For mpi support, you need to define macro `TAT_USE_MPI` and use mpi compiler such as `mpic++`(recommend) or pass correct flag to normal compiler(for expert)

Please check comment in file [`TAT.hpp`](/include/TAT/TAT.hpp#L42) for some other macro option

Please notice that this library need proper compiler optimization option(`-O2`, `-O3`, `-Ofast`) for good performace

## Python binding
Python binding is configured in [`CMakeLists.txt`](/CMakeLists.txt#L127), use cmake and build target `TAT`

For other customed python module name, define `TAT_PYTHON_MODULE` as cmake variable

If you are familiar with [pybind11](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually), you can compile [`python/TAT.cpp`](/python/TAT.cpp) directly with correct flag

Please notice that for some old mpi version(for example, openmpi 2.1.1 in ubuntu 18.04 LTS), you need to load mpi dynamic shared library manually before `import TAT`, The way to load it manually is `import ctypes` and `ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)`

## Use with [emscripten](https://emscripten.org/)
If you want to run a program using TAT in browser, which is very useful for demonstration.

You can simply compile TAT with `em++`(no mpi support, no doubt), and link `libblas.a`, `libf2c.a`, `liblapack.a` compiled from [clapack-3.2.1](https://www.netlib.org/clapack/)

You can download them from [here](https://github.com/hzhangxyz/TAT/releases/tag/v0.0.6) or compile by yourself.
