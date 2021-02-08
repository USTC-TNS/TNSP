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

No Symmetry Tensor is a tensor without any symmetry,
which is just a normal tensor, so let us begin from this
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

```c++
auto A = Tensor({"i"}, {10}).test();
// Get maximum norm
std::cout << A.norm<-1>() <<"\n";
// Get 0 norm
std::cout << A.norm<0>() <<"\n";
// Get 1 norm
std::cout << A.norm<1>() <<"\n";
// Get 2 norm
std::cout << A.norm<2>() <<"\n";
```

#### Contract

```c++
auto A = Tensor({"i", "j", "k"}, {2, 3, 4}).test();
auto B = Tensor({"a", "b", "c", "d"}, {2, 5, 3, 6}).test();
// Contract edge i of A and edge a of B, edge j of A and edge c of B
auto C = A.contract(B, {{"i", "a"}, {"j", "c"}});
std::cout << C << "\n";
```

#### Merge and split edge

```c++
auto A = Tensor({"i", "j", "k"}, {2, 3, 4}).test();
// Merge edge i and edge j into a single edge a,
// and Merge no edge to get a trivial edge b
auto B = A.merge_edge({{"a", {"i", "j"}}, {"b", {}}});
std::cout << B << "\n";

// Split edge a back to edge i and edge j, and split
// trivial edge b to no edge
auto C = B.split_edge({{"b", {}}, {"a", {{"i", 2}, {"j", 3}}}});
std::cout << C << "\n";
```

#### Edge rename and transpose

```c++
auto A = Tensor({"i", "j", "k"}, {2, 3, 4}).test();
// Rename edge i to edge x
auto B = A.edge_rename({{"i", "x"}});
std::cout << B << "\n";
// `edge_rename` is an outplace operator
std::cout << A << "\n";

// Transpose tensor A with specific order
auto C = A.transpose({"k", "j", "i"});
std::cout << C << "\n";
```

#### SVD and QR decomposition

##### QR decomposition

```c++
auto A = Tensor({"i", "j", "k"}, {2, 3, 4}).test();
// Do QR decomposition, specify Q matrix edge is edge k
// You can also write is as `Q, R = A.qr('r', {"i", "j"}, "Q", "R")`
// The last two argument is the name of new edges generated
// by QR decomposition
auto [Q, R] = A.qr('q', {"k"}, "Q", "R");
// Q is an unitary matrix, which edge name is Q and k
std::cout << Q.edge_rename({{"Q", "Q1"}}).contract(Q.edge_rename({{"Q", "Q2"}}), {{"k", "k"}}) << "\n";
// Q R - A is 0
std::cout << (Q.contract(R, {{"Q", "R"}}) - A).norm<-1>() << "\n";
```

##### SVD decomposition

```c++
// Do SVD decomposition with cut=3, if cut not specified,
// svd will not cut the edge.
// The first argument is edge set of matrix U, SVD does not
// supply function to specify edge set of matrix V like what
// is done in QR since SVD is symmetric between U and V.
// The later two argument is new edges generated
auto [U, S, V] = A.svd({"k"}, "U", "V", 3);
// U is an rank-3 unitary matrix
std::cout << U.edge_rename({{"U", "U1"}}).contract(U.edge_rename({{"U", "U2"}}), {{"k", "k"}}) << "\n";
// U S V - A is a small value
// please notice that S is an diagnalized matrix so contract is
// not support, use multiple which is designed for this
// situation instead. Its interface is
// `matrix_U.multiple(Singular, matrix_U_edge_name, 'u')` or
// `matrix_V.multiple(Singular, matrix_V_edge_name, 'v')`,
// multiple is an outplace operator
std::cout << (U.multiple(S, "U", 'u').contract(V, {{"U", "V"}}) - A).norm<-1>() << "\n";

// Here A is a real tensor, if it is complex tensor, you may
// need outplace operator `U.conjugate()` to get conjugate
// tensor of unitary matrix
```

#### Identity, exponential and trace

```c++
// Please notice that identity is INPLACE operator
// For any i, j, k, l, we have
// `A[{"i":i, "j":j, "k":k, "l":l}] = delta(i,l) * delta(j,k)`
auto A = Tensor({"i","j","k","l"},{2,3,3,2}).identity({{"i", "l"}, {"j", "k"}});

// calculate matrix exponential B = exp(A)
// second argument is iteration steps, with default value 2
auto B = A.exponential({{"i", "l"}, {"j", "k"}}, 4);

// Calculate trace or partial trace of a tenso
// Here it calculate `A[{"i":i, "j":j, "k":k, "l":l}] * delta(i,l) * delta(j,k)`
auto C = A.trace({{"i", "l"}, {"j", "k"}});
```

#### IO

You can direclty read/write/load/dump tensor from/to a stream.

```c++
auto A = Tensor({"i","j","k","l"},{2,3,3,2}).identity({{"i", "l"}, {"j", "k"}});
std::stringstream text_stream;
// use operator<< to write to a stream
text_stream << A;
std::cout << text_stream.str() << "\n";
Tensor B;
// use operatoor>> to read from a stream
text_stream >> B;

std::stringstream binary_stream;
// use operator< to dump to a stream
binary_stream < A;
Tensor C;
// use operator> to load from a stream
binary_stream > C;
```

#### Fill random number into tensor

c++ have its own way to generate random number, see [this](https://en.cppreference.com/w/cpp/numeric/random).
So TAT will use this to generate random tensor.

`Tensor::set` is an inplace operator with one function as its argument,
its will call this function to get every element of the tensor.
It will be used to get random tensor with help of c++ own random library.

```c++
std::random_device rd;
auto seed = rd();
std::default_random_engine engine(seed);
std::normal_distribution<double> dist{0, 1};
auto A = Tensor({"i", "j", "k"}, {2, 3, 4}).set([&](){ return dist(engine); });
std::cout << A << "\n";
```

### Symmetry Tensor

### Fermi Tensor

## Links

- [a tensor network library](https://github.com/crimestop/net)
- [gitee mirror of TAT](https://gitee.com/hzhangxyz/TAT)
