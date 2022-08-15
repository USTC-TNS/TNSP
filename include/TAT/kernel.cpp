#include <complex>
#include <cstdint>
#include <thrust/complex.h>

__global__ void line_copy_kernel_parity_true(
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_number,
      std::uint64_t line_size_value) {
   auto index = blockIdx.x * blockDim.x + threadIdx.x;
   auto line = index / line_size_value;
   auto i = index % line_size_value;
   const thrust::complex<double>* __restrict source = reinterpret_cast<const thrust::complex<double>*>(source_lines[line]);
   thrust::complex<double>* __restrict destination = reinterpret_cast<thrust::complex<double>*>(destination_lines[line]);
   if (line < line_number)
      destination[i] = -source[i];
}
__global__ void line_copy_kernel_parity_false(
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_number,
      std::uint64_t line_size_value) {
   auto index = blockIdx.x * blockDim.x + threadIdx.x;
   auto line = index / line_size_value;
   auto i = index % line_size_value;
   const thrust::complex<double>* __restrict source = reinterpret_cast<const thrust::complex<double>*>(source_lines[line]);
   thrust::complex<double>* __restrict destination = reinterpret_cast<thrust::complex<double>*>(destination_lines[line]);
   if (line < line_number)
      destination[i] = source[i];
}

void line_copy_interface(
      std::uint64_t line_number,
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_size_value,
      bool parity) {
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   auto block_number = line_number * line_size_value / 32 + 1;
   if (parity)
      line_copy_kernel_parity_true<<<block_number, 32, 0, stream>>>(source_lines, destination_lines, line_number, line_size_value);
   else
      line_copy_kernel_parity_true<<<block_number, 32, 0, stream>>>(source_lines, destination_lines, line_number, line_size_value);
   void** userData = new void*[2];
   userData[0] = source_lines;
   userData[1] = destination_lines;
   cudaStreamAddCallback(
         stream,
         [](cudaStream_t stream, cudaError_t status, void* userData_) {
            void** userData = reinterpret_cast<void**>(userData_);
            cudaFree(userData[0]);
            cudaFree(userData[1]);
            delete[] userData;
         },
         userData,
         0);
   cudaStreamDestroy(stream);
}
