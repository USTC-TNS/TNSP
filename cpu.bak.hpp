#ifndef TENSOR_CPU_HPP_
#define TENSOR_CPU_HPP_

#include "meta.hpp"

#include "eigen-git-mirror/Eigen/Dense"
#include "eigen-git-mirror/unsupported/Eigen/CXX11/Tensor"

namespace Node
{
  namespace internal
  {
    namespace memory
    {
      inline void deleter::operator()(Base* ptr) const
      {
        delete[] ptr;
      }

      std::unique_ptr<Base[], deleter> newer(Size size)
      {
        return std::unique_ptr<Base[], deleter>(new Base[size]);
      }

      inline void memCopy(void* dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }

      inline void memSend(void*dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }

      inline void memRecv(void* dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }
    }

    namespace shuffle
    {
      template<Rank N>
      void eigen_shuffle(PlainData                                   data_new,
                         PlainData                                   data_old,
                         const Dims&                            dims_new,
                         const Dims&                            dims_old,
                         const Order&                           plan)
      {
        Eigen::array<Size, N> arr_new, arr_old;
        //Eigen::array<Rank, N> arr_plan;
        std::copy(dims_new.begin(), dims_new.end(), arr_new.begin());
        std::copy(dims_old.begin(), dims_old.end(), arr_old.begin());
        //std::copy(plan.begin(), plan.end(), arr_plan.begin());
        /*for(Rank i=0;i<N;i++)
          {
          arr_new[i] = dims_new[i];
          arr_old[i] = dims_old[i];
          //arr_plan[i] = plan[i];
          }*/
        Eigen::TensorMap<Eigen::Tensor<Base, N, Eigen::RowMajor>> tensor_new(data_new, arr_new);
        Eigen::TensorMap<Eigen::Tensor<Base, N, Eigen::RowMajor>> tensor_old(data_old, arr_old);
        tensor_new = tensor_old.shuffle(plan);
      }

      using ShuffleType = decltype(eigen_shuffle<0>);
      static ShuffleType* shuffle_list[] = {eigen_shuffle<0>, eigen_shuffle<1>, eigen_shuffle<2>, eigen_shuffle<3>, eigen_shuffle<4>, eigen_shuffle<5>, eigen_shuffle<6>, eigen_shuffle<7>,
                                            eigen_shuffle<8>, eigen_shuffle<9>, eigen_shuffle<10>, eigen_shuffle<11>, eigen_shuffle<12>, eigen_shuffle<13>, eigen_shuffle<14>, eigen_shuffle<15>};

      void shuffle(PlainData    data_new,
                   PlainData    data_old,
                   const Dims&  dims_new,
                   const Dims&  dims_old,
                   const Order& plan)
      {
        shuffle_list[plan.size()](data_new, data_old, dims_new, dims_old, plan);
      }
    }

    /*
      namespace contract
      {
      template<>
      void gemm<double>(double*                                data,
      double*                                data1,
      double*                                data2,
      Size                                   a,
      Size                                   b,
      Size                                   c)
      {
      Eigen::Map<Eigen::Matrix<Base, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix1(data1, a, b);
      Eigen::Map<Eigen::Matrix<Base, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix2(data2, b, c);
      Eigen::Map<Eigen::Matrix<Base, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(data, a, c);
      matrix = matrix1 * matrix2;
      }
      }
    */
  }
}

#endif
