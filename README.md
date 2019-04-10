# [TAT](https://github.com/hzhangxyz/TAT) &middot; [![version](https://img.shields.io/github/release/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/releases/latest) [![license](https://img.shields.io/github/license/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/blob/TAT/LICENSE) ![platform](https://img.shields.io/badge/platform-linux-brightgreen.svg) ![build](https://travis-ci.com/hzhangxyz/TAT.svg?branch=TAT)

TAT is A Tensor library

## current function
- a TAT lib
- a MPS demo

## dependence

|Repo|License|
|----|-------|
| [![dependence](https://img.shields.io/badge/Taywee-args-blue.svg)](https://github.com/Taywee/args) | [![license](https://img.shields.io/github/license/Taywee/args.svg)](https://github.com/Taywee/args/blob/master/LICENSE) |
| [![dependence](https://img.shields.io/badge/springer13-hptt-blue.svg)](https://github.com/springer13/hptt) | [![license](https://img.shields.io/github/license/springer13/hptt.svg)](https://github.com/springer13/hptt/blob/master/LICENSE.txt) |
| [![dependence](https://img.shields.io/badge/agauniyal-rang-blue.svg)](https://github.com/agauniyal/rang) | [![license](https://img.shields.io/github/license/agauniyal/rang.svg)](https://github.com/agauniyal/rang/blob/master/LICENSE) |

## history
after many try and exercise and rewrite include

|Repo|Time|Desciption|
|----|----|----------|
| [![history](https://img.shields.io/badge/MPS-np/tf-red.svg)](https://github.com/Aaaaaaaah/MPS) | 2017 Autumn | MPS implementation in np |
| [![history](https://img.shields.io/badge/SquareLattice-np-red.svg)](https://github.com/Aaaaaaaah/SquareLattice) | 2018 Spring | attempt to write a library, failure |
| [![history](https://img.shields.io/badge/tnsp-np/tf-red.svg)](https://github.com/hzhangxyz/tnsp) | 2018 Autumn | tensorflow wrap as tensor library |
| [![history](https://img.shields.io/badge/TNC-Eigen-red.svg)](https://github.com/hzhangxyz/TNC) | 2019 Winter | eigen wrap as tensor library |

and with practice and experience of using [![history](https://img.shields.io/badge/TNSP-Fortran-blue.svg)](https://arxiv.org/pdf/1708.00136.pdf)

now, I am writing this tensor library.

---
The following is for developer to take note

## TODO LIST
- transpose may be optimized
- Truncated SVD
- dgegqr magma
- cuda，sw
- site api
- use it, peps, kitaev, hubbard
- symmetry tensor

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
