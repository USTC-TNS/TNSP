# TAT
TAT is A Tensor library

## current function
- a [TAT lib](https://github.com/hzhangxyz/TAT/blob/TAT/TAT.hpp)
- a [MPS demo](https://github.com/hzhangxyz/TAT/blob/TAT/Heisenberg_MPS.cpp)

## history
after many try and exercise and rewrite include

- [MPS-np/tf](https://github.com/Aaaaaaaah/MPS) 2017 Autumn
- [SquareLattice-np/tf](https://github.com/Aaaaaaaah/SquareLattice) 2018 Spring
- [tnsp-np/tf](https://github.com/hzhangxyz/tnsp) 2018 Autumn
- [TNC-eigen](https://github.com/hzhangxyz/TNC) 2019 Winter

now, I am writing this tensor library finally.

## dependence
- [args](https://github.com/Taywee/args) MIT License
- [hptt](https://github.com/springer13/hptt) BSD 3-Clause License

---

## TODO LIST
- transpose may be optimized
- Truncated SVD
- dgegqr magma
- cuda，sw
- use it, peps, kitaev, hubbard

### note for some of cuda program
```c++
namespace shuffle
{
  void shuffle(PlainData    data_new,
               PlainData    data_old,
               const Dims&  dims_new,
               const Dims&  dims_old,
               const Order& plan)
  {
    //Stream !!!
    const Rank& size = plan.size();
    std::vector<int> int_plan(size, 0);//(plan.begin(), plan.end());
    std::vector<int> int_dims(size, 0);//(dims_old.begin(), dims_old.end());
    for(Rank i=0;i<size;i++)
      {
        int_plan[i] = size - plan[size-i-1] -1;
        int_dims[i] = dims_old[size-i-1];
        //std::cout << plan[i] << "\t" << int_plan[i] << "\t" << dims_old[i] << "\t" << int_dims[i] << "\n";
      }
    //std::cout << "\n\n\n";
    cuttHandle handle;
    internal::cuda::Stream* stream = internal::cuda::get_stream();
    cuttPlan(&handle, size, int_dims.data(), int_plan.data(), sizeof(Base), stream->stream);
    cuttExecute(handle, data_old, data_new);
    cudaStreamSynchronize(stream->stream);
    internal::cuda::delete_stream(stream);
  }
}

namespace contract
{
  template<>
  void gemm<double>(double* data,
                    double* data1,
                    double* data2,
                    Size    a,
                    Size    b,
                    Size    c)
  {
    double alpha = 1;
    double beta  = 0;
    internal::cuda::Stream* stream = internal::cuda::get_stream();
    cublasDgemm(stream->handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &alpha, data2, c, data1, b, &beta, data, c);
    cudaStreamSynchronize(stream->stream);
    internal::cuda::delete_stream(stream);
  }
}
```
