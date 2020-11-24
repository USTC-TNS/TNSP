# No Symmetry Tensor

No Symmetry Tensor is a tensor without any symmetry, which is just a normal tensor. In `TAT`, use
```{cpp}
using Tensor = TAT::Tensor<double, TAT::NoSymmetry, TAT::DefaultName>;
```
to get `Tensor` as no symmetry tensor with basic scalar type as `double`.

In fact, those are `TAT::Tensor`'s default template argument, so you can directly use 
```{cpp}
using Tensor = TAT::Tensor<>;
```

For `TAT::Tensor`, the first argument is tensor's basic scalar type, the second is the symmetry the tensor has, and the third is the name type to mark tensor's index.

# Create tensor {#create-tensor}
To create a no symmetry tensor, pass names and dimension for each dimension of the tensor
```{cpp}
auto tensor = TAT::Tensor<double, TAT::NoSymmetry>({"A", "B"}, {3, 4});
std::cout << tensor1 << "\n";
```
The result is something like
```{cpp}
{names:[A,B],edges:[3,4],blocks:[6.91832e-310,4.66918e-310,3.16202e-322,3.95253e-322,4.66986e-310,0,3.39519e-313,5.76075e+180,4.1237e+97,2.69151e+276,1.39981e+93,1.19745e+16]}
```
Please notice that the tensor content is not initialized yet. To initialize it, see [next section](#initialize-tensor)

When you pass names information and dimension information, You cannot pass type `std::vector<std::string>` or `std::vector<int>`, unless you define macro `TAT_USE_EASY_CONVERSION` before include `TAT/TAT.hpp`. if this macro not defined, the argument type is `std::vector<TAT::FastName>` and `std::vector<TAT::Edge<TAT::NoSymmetry>>`, and you need to do conversion manually. Since there are implicit conversion from `std::string` to `TAT::FastName`, you can write something like
```{cpp}
std::vector<std::string> name_list = ...;
TAT::Tensor<>({name_list.begin(), name_list.end()}, ...);
```
And the similar way also works for `TAT::Edge<TAT::NoSymmetry>`.

`TAT::Tensor` is rebust for zero condition, you can run
```{cpp}
std::cout << TAT::Tensor<>({}, {}) << "\n";
```
to get rank 0 tensor and result is something like
```
{names:[],edges:[],blocks:[4.67467e-310]}
```
you can also run
```{cpp}
std::cout << TAT::Tensor<>({"A","B","C"}, {2333, 0, 2333}) << "\n";
```
to get a tensor with one dimension equals to zero and the result is
```
{names:[A,B,C],edges:[2333,0,2333],blocks:[]}
```


# Initialize tensor {#initialize-tensor}

## Initialize tensor data for test
You can easily initialize data for some simple test by `tensor.test()`, which will initialize data as natural number sequence. For example
```{cpp}
auto tensor = TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C"}, {2, 2, 3}).test();
std::cout << tensor << "\n";
```
The result is
```
{names:[A,B,C],edges:[2,2,3],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}
```

## Initialize tensor's elements to zero
To initialize all tensor's elements to zero, call `tensor.zero()`, for example
```{cpp}
auto tensor = TAT::Tensor<double, TAT::NoSymmetry>({"X", "Y", "Z"}, {1, 2, 3}).zero();
std::cout << tensor << "\n";
```
The result is
```
{names:[X,Y,Z],edges:[1,2,3],blocks:[0,0,0,0,0,0]}
```