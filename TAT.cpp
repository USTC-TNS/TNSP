#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <future>
#include <memory>
#include <cstdlib>
#include <cstring>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ <<std::endl;

enum class Device {CPU, CUDA, DCU, SW};

enum class Legs
  {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
   CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4),
   CreateLeg(5), CreateLeg(6), CreateLeg(7), CreateLeg(8), CreateLeg(9)
#undef CreateLeg
  };

#define DefineLeg(x) static const Legs x = Legs::x
#define DefineLegs(n) DefineLeg(Left##n); DefineLeg(Right##n); DefineLeg(Up##n); DefineLeg(Down##n); DefineLeg(Phy##n)
DefineLegs(); DefineLegs(1); DefineLegs(2); DefineLegs(3); DefineLegs(4);
DefineLegs(5); DefineLegs(6); DefineLegs(7); DefineLegs(8); DefineLegs(9);
#undef DefineLegs
#undef DefineLeg

#define IncEnum(p) {Legs::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
static const std::map<Legs, std::string> legs_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
                                                   IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)};
#undef IncGroup
#undef IncEnum

std::ostream& operator<<(std::ostream& out, const Legs& value)
{
  try
    {
      return out << legs_str.at(value);
    }
  catch(const std::out_of_range& e)
    {
      return out;
    }
}

using Size = std::size_t;
using Rank = unsigned int;

template<Device device, class Base>
class Data;

template<class Base>
class Data<Device::CPU, Base>{
public:
  Size size;
  std::unique_ptr<Base[]> data;

  Data() = delete;
  ~Data() = default;
  Data(Data<Device::CPU, Base>&& other) = default;
  Data<Device::CPU, Base>& operator=(Data<Device::CPU, Base>&& other) = default;
  Data(Size _size) : size(_size) {
    data = std::unique_ptr<Base[]>(new Base[size]);
  }
  Data(const Data<Device::CPU, Base>& other){
    new (this) Data(other.size);
    std::memcpy(data.get(), other.data.get(), size*sizeof(Base));
  }
  Data<Device::CPU, Base>& operator=(const Data<Device::CPU, Base>& other){
    new (this) Data(other);
  }
};

template<Device device, class Base>
class Node{
public:
  Rank rank;
  std::vector<Size> dims;
  Data<device, Base> data;

  Node() = delete;
  ~Node() = default;
  Node(Node<device, Base>&& other) = default;
  Node(const Node<device, Base>& other) = default;
  Node<device, Base>& operator=(Node<device, Base>&& other) = default;
  Node<device, Base>& operator=(const Node<device, Base>& other) = default;
  static Size get_size(const std::vector<Size>& _dims){
    Size res = 1;
    for(auto i : _dims){
      res *= i;
    }
    return res;
  }
  template<class T=std::vector<Size>>
  Node(T&& _dims) : data(get_size(_dims)), rank(_dims.size()), dims(std::forward<T>(_dims)){}
};

template<Device device=Device::CPU, class Base=double>
class Tensor{
public:
  Rank rank;
  std::vector<Legs> legs;
  Node<device, Base> node;

  Tensor() = delete;
  ~Tensor() = default;
  Tensor(Tensor<device, Base>&& other) = default;
  Tensor(const Tensor<device, Base>& other) = default;
  Tensor<device, Base>& operator=(Tensor<device, Base>&& other) = default;
  Tensor<device, Base>& operator=(const Tensor<device, Base>& other) = default;
  template<class T1=std::vector<Size>, class T2=std::vector<Legs>>
  Tensor(T1&& _dims, T2&& _legs) : node(std::forward<T1>(_dims)), rank(_legs.size()), legs(std::forward<T2>(_legs)){}
};

int main(){
  Tensor<> t1({2,3},{Up, Down});
  return 0;
}
