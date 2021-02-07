# [TAT](https://github.com/hzhangxyz/TAT) &middot; [![version](https://img.shields.io/github/v/tag/hzhangxyz/TAT?style=flat-square)](https://github.com/hzhangxyz/TAT/tags) [![license](https://img.shields.io/github/license/hzhangxyz/TAT?style=flat-square)](/LICENSE.md) [![build](https://img.shields.io/github/workflow/status/hzhangxyz/TAT/check?style=flat-square)](https://github.com/hzhangxyz/TAT/actions?query=workflow%3Acheck) [![document](https://img.shields.io/github/workflow/status/hzhangxyz/TAT/doxygen?color=%237f7fff&label=doc&style=flat-square)](https://hzhangxyz.github.io/TAT/index.html)

TAT is a header-only c++ tensor library with support for Abelian [symmetry tensor](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.050301) and [Fermi tensor](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.195153)

The name "TAT" is a recursive acronym for "TAT is A Tensor library!", and it should be all uppercase

## Prerequisites

- c++ compiler with c++17 support(such as gcc7+, clang5+, msvc19.14+)
- lapack/blas or mkl
- mpi(optional for parallel computing)
- [pybind11](https://github.com/pybind/pybind11)(optional for python binding)
- [numpy](https://github.com/numpy/numpy)(optional in python binding to export data into numpy array)

There is also some dependents library used by some of examples/demos/tests.

- [fire-hpp](https://github.com/kongaskristjan/fire-hpp)
- [fire](https://github.com/google/python-fire)
- [multimethod](https://github.com/coady/multimethod)

## Usage

Just include the file [`include/TAT/TAT.hpp`](/include/TAT/TAT.hpp) and link lapack/blas or mkl at link time

For good practice, pass argument `-I$path_to_TAT_root/include` to compiler and use `#include <TAT/TAT.hpp>` in your source file

For mpi support, you need to define macro `TAT_USE_MPI` and use mpi compiler such as `mpic++`(recommend) or pass correct flag to normal compiler(for expert)

Please check comment in file [`TAT.hpp`](/include/TAT/TAT.hpp#L42) for some other macro options

Please notice that this library need proper compiler optimization option(`-O2`, `-O3`, `-Ofast`) for good performace

You can also use TAT as a cmake subdirectory, just use `add_subdirectory(path_to_TAT_root)` and `target_link_libraries(your_target TAT)` in your `CMakeLists.txt`

## Python binding

Python binding is configured in [`CMakeLists.txt`](/CMakeLists.txt#L117), use cmake and build target `PyTAT`

For other customed python module name, define `TAT_PYTHON_MODULE` as cmake variable

If you are familiar with [pybind11](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually), you can compile [`python/TAT.cpp`](/python/TAT.cpp) directly with correct flag

Refer to [this](/PyTAT/README.md) for document of python binding

## Use with [emscripten](https://emscripten.org/)

If you want to run a program using TAT in browser, which is very useful for demonstration

You can simply compile TAT with `em++`(no mpi support, no doubt), and link `liblapack.a`, `libblas.a`, `libf2c.a` compiled from [clapack-3.2.1](https://www.netlib.org/clapack/)

You can download them from [here](https://github.com/hzhangxyz/TAT/releases/tag/v0.0.6) or compile by yourself

If you are using cmake, you need to put these three files into directory `emscripten`, then run `emcmake cmake $path_to_TAT_root` which will configure it automatically

## Documents

### No Symmetry Tensor

No Symmetry Tensor is a tensor without any symmetry, which is just a normal tensor, so let us begin from this
In `TAT`, use

```c++
using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;
```

to get `Tensor` as no symmetry tensor with basic scalar type as `double`.

#### Create tensor

To create a no symmetry tensor, pass names and dimension for each dimension of the tensor

```c++
auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i", "j"}, {3, 4});
std::cout << A << "\n";
```

the code above create a rank-2 tensor named `A` which two edges are `i` and `j`,
and their dimensions are `3` and `4`, then print tensor `A` to `std::cout`.

Please notice that TAT will NOT initialize content of tensor when create it.

#### Access element of tensor

You can easily access elements of tensor by a map from name of edge to index

```c++
// Create a tensor and initialize it to zero
auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i", "j"}, {3, 4}).zero();
// Set an element of tensor A to 3
A.at({{"i", 2}, {"j", 2}}) = 3;
// print tensor A
std::cout << A << "\n";
```

#### Scalar operators

You can do scalar operators directly

```c++
// Create two rank-1 tensors
auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i"}, {4});
auto B = TAT::Tensor<double, TAT::NoSymmetry>({"i"}, {4});
// A kind of low level value setter, directly set array of tensor content
A.block() = {1, 2, 3, 4};
B.block() = {10, 20, 30, 40};

// Add two tensor
std::cout << A + B << "\n";

// A number over a tensor
std::cout << 1 / A << "\n";
```

#### Rank-0 tensor and number

You can convert between rank-0 tensor and number directly

```c++
// Directly initialize a tensor with a number
auto A = TAT::Tensor<double, TAT::NoSymmetry>(233);

// Convert rank-0 tensor to number
double a = A;
```

#### Explicitly copy

```c++
auto A = TAT::Tensor<double, TAT::NoSymmetry>(233);
// By default, assigning a tensor to another tensor
// will let two tensor share the same data blocks
auto B = A;
// data of B is not changed when execute `A.at({}) = 1`
// but data copy happened implicitly and a warning will
// be thrown.
A.at({}) = 1;

auto C = TAT::Tensor<double, TAT::NoSymmetry>(233);
// Explicitly copy of tensor C
auto D = C.copy();
// No warning will be thrown
C.at({}) = 1;
```

#### Create same shape tensor

```c++
auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i", "j"}, {2, 2});
A.block() = {1, 2, 3, 4};
// tensor B copy the shape of A but not content of A
auto B = A.same_shape();
std::cout << B << "\n";
```

#### Map and transform

```c++
using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;
auto A = Tensor({"i", "j"}, {2, 2});
// Another easy test data setter for tensor
// which will fill meanless test data into tensor
A.test();
// Every element is transformed by a function inplacely
A.transform([](auto x){ return x * x; });
std::cout << A << "\n";

// Every element is transformed by a function outplacely
auto B = A.map([](auto x){ return x + 1; });
std::cout << B << "\n";
std::cout << A << "\n";
```

#### Type conversion

```c++
// decltype(A) is TAT::Tensor<double, TAT::NoSymmetry>
auto A = TAT::Tensor<double, TAT::NoSymmetry>(233);
// Convert A to an complex tensor
// decltype(B) is  TAT::Tensor<std::complex<double>, TAT::NoSymmetry>
auto B = A.to<std::complex<double>>();
```

#### Norm

#### Contract

#### Merge and split edge

#### Edge rename and transpose

#### SVD and QR decomposition

#### Identity, exponential and trace

#### IO

#### Fill random number into tensor

## Links

- [a tensor network library](https://github.com/crimestop/net)
- [gitee mirror of TAT](https://gitee.com/hzhangxyz/TAT)
