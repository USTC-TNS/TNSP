#include <cstdlib>
#include <cstring>

namespace Node
{
  using Size = std::size_t;
  using Base = double;
  enum class Leg
  {
    #define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
    CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
    #undef CreateLeg
  };

  enum class Device
  {
  CUDA, SW, AMD, CPU
  };

  enum class Where
  {
    Host, Device
  };

  namespace internal::memory
  {
    // run in host, malloc in device
    template<Device host, Device device>
    void malloc(void**, std::size_t);

    template<Device host, Device device>
    void free(void*);

    template<Device host, Device device>
    void memcpy(void*, const void*, std::size_t);

    template<>
    void malloc<Device::CPU, Device::CPU>(void** ptr, std::size_t size)
    {
      *ptr = std::malloc(size);
    }

    template<>
    void free<Device::CPU, Device::CPU>(void* ptr)
    {
      std::free(ptr);
    }

    template<>
    void memcpy<Device::CPU, Device::CPU>(void* dst, const void* src, std::size_t size)
    {
      std::memcpy(dst, src, size);
    }
  }


  namespace internal::stream
  {
    namespace internal
    {
      template<Device device>
      class stream_aux
      {
      };

      template<>
      class stream_aux<Device::SW>
      {
        using stream = int;
      };
    }

    template<Device device>
    using stream = typename internal::stream_aux<device>::stream;
  }

  namespace internal::shuffle
  {
    void get_plan(Size*& plan, Leg* legs_old, Leg* legs_new);
  
    template<Device device>
    void shuffle();
  }

  // where means where data dims, legs save, data is always saved in device
  template<Device device, Where where>
  class Tensor
  {
    public:
      Size  rank;
      Size* dims;
      Leg*  legs;
      Base* data;
      Size  size;
      static const Device data_device = device;
      static const Device meta_device = (where==Where::Host)?Device::CPU:device;
    private:
      void malloc(void** ptr, std::size_t size)
      {
        internal::memory::malloc<meta_device, meta_device>(ptr, size);
      }
      void malloc_data(void** ptr, std::size_t size)
      {
        internal::memory::malloc<meta_device, device>(ptr, size);
      }
      void free(void* ptr)
      {
        internal::memory::free<meta_device, meta_device>(ptr, size);
      }
      void free_data(void* ptr)
      {
        internal::memory::free<meta_device, device>(ptr, size);
      }
      void memcpy(void* dst, const void* src, std::size_t size)
      {
        internal::memory::memcpy<meta_device, meta_device>(dst, src, size);
      }
      void memcpy_data(void* dst, const void* src, std::size_t size)
      {
        internal::memory::memcpy<meta_device, device>(dst, src, size);
      }
      void free_all()
      {
        if(dims) free(dims);
        if(legs) free(legs);
        if(data) free_data(data);
      }
      void init()
      {
        rank = 0;
        dims = nullptr;
        legs = nullptr;
        data = nullptr;
        size = 1;
      }
    public:
      void clean()
      {
        free_all();
        init();
      }
      Tensor()
      {
        init();
      }
      ~Tensor()
      {
        free_all();
      }
      void shuffle_to(Tensor<device, where>& tensor, Leg* new_legs, internal::stream::stream<device> stream)
      {
        tensor.clean();
        malloc(&tensor.dims, sizeof(Size)*rank);
        malloc(&tensor.legs, sizeof(Leg)*rank);
        malloc_data(&tensor.data, sizeof(Base)*size);
        tensor.rank = rank;
        tensor.size = size;
        Size* plan;
        malloc(&plan, sizeof(Size)*rank);
        internal::shuffle::get_plan(plan, legs, new_legs);
        internal::shuffle::shuffle<device>();
      }
  };
}