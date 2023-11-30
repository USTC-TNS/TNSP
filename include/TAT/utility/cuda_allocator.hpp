#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

namespace TAT {
    namespace cuda {
        template<typename T>
        struct allocator : std::allocator<T> {
            using std::allocator<T>::allocator;

            allocator(const allocator& other) = default;
            template<typename U>
            allocator(const allocator<U>& other) noexcept : allocator() { }

            allocator<T> select_on_container_copy_construction() const {
                return allocator<T>();
            }

            // It is useless, but base class has it so derived class must have it.
            template<typename U>
            struct rebind {
                using other = allocator<U>;
            };

            T* allocate(std::size_t n) {
                T* p = nullptr;
                auto code = cudaMallocManaged(&p, n * sizeof(T));
                if (code != cudaSuccess) {
                    printf("GPU assert: %s\n", cudaGetErrorString(code));
                }
                return p;
            }
            void deallocate(T* p, std::size_t) {
                cudaFree(p);
            }

            template<typename U, typename... Args>
            void construct([[maybe_unused]] U* p, Args&&... args) {
                if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
                    new (p) U(std::forward<Args>(args)...);
                }
            }
        };
        template<typename T>
        using vector = std::vector<T, allocator<T>>;
    } // namespace cuda
} // namespace TAT
