#include <complex>
#include <cstdint>
#include <thrust/complex.h>

__global__ void line_copy_kernel(
      std::uint64_t line,
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_size_value,
      bool parity) {
   const thrust::complex<double>* __restrict source = reinterpret_cast<const thrust::complex<double>*>(source_lines[line]);
   thrust::complex<double>* __restrict destination = reinterpret_cast<thrust::complex<double>*>(destination_lines[line]);
   for (std::uint64_t i = 0; i < line_size_value; i++) {
      if (parity) {
         destination[i] = -source[i];
      } else {
         destination[i] = source[i];
      }
   }
}

void line_copy_interface(
      std::uint64_t line_number,
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_size_value,
      bool parity) {
   for (std::uint64_t line = 0; line < line_number; line++) {
      line_copy_kernel<<<1, 1>>>(line, source_lines, destination_lines, line_size_value, parity);
   }
   cudaDeviceSynchronize();
}
