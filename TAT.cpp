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

namespace legs{
  enum class Legs
    {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
     CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4),
     CreateLeg(5), CreateLeg(6), CreateLeg(7), CreateLeg(8), CreateLeg(9)
#undef CreateLeg
    };

#define IncEnum(p) {Legs::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
  static const std::map<Legs, std::string> legs_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
                                                       IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)};
#undef IncGroup
#undef IncEnum

  std::ostream& operator<<(std::ostream& out, const Legs& value){
    try{
        return out << legs_str.at(value);
    } catch(const std::out_of_range& e) {
        return out;
    }
  }
}
using legs::Legs;

#define DefineLeg(x) static const Legs x = Legs::x
#define DefineLegs(n) DefineLeg(Left##n); DefineLeg(Right##n); DefineLeg(Up##n); DefineLeg(Down##n); DefineLeg(Phy##n)
DefineLegs(); DefineLegs(1); DefineLegs(2); DefineLegs(3); DefineLegs(4);
DefineLegs(5); DefineLegs(6); DefineLegs(7); DefineLegs(8); DefineLegs(9);
#undef DefineLegs
#undef DefineLeg


using Size = std::size_t;
using Rank = unsigned int;

namespace data{
  template<Device device, class Base>
  class Data;

  template<class Base>
  class Data<Device::CPU, Base>{
  public:
    Size size;
    std::unique_ptr<Base[]> base;

    Data() = delete;
    ~Data() = default;
    Data(Data<Device::CPU, Base>&& other) = default;
    Data<Device::CPU, Base>& operator=(Data<Device::CPU, Base>&& other) = default;
    Data(Size _size) : size(_size) {
      base = std::unique_ptr<Base[]>(new Base[size]);
    }
    Data(const Data<Device::CPU, Base>& other){
      new (this) Data(other.size);
      std::memcpy(base.get(), other.base.get(), size*sizeof(Base));
    }
    Data<Device::CPU, Base>& operator=(const Data<Device::CPU, Base>& other){
      new (this) Data(other);
    }

    void generate_test(){
      for(Size i=0;i<size;i++){
        base[i] = i;
      }
    }
  };

  template<Device device, class Base>
  std::ostream& operator<<(std::ostream& out, const Data<device, Base>& value){
    for(Size i=0;i<value.size-1;i++){
      out << value.base[i] << " ";
    }
    if(value.size!=0){
      out << value.base[value.size-1];
    }
    return out;
  }
}
using data::Data;

namespace node{
  template<Device device, class Base>
  class Node{
  public:
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
    Node(T&& _dims) : data(get_size(_dims)){
      dims = std::forward<T>(_dims);
    }

    void generate_test(){
      data.generate_test();
    }
  };

  std::ostream& operator<<(std::ostream& out, const std::vector<Size>& value){
    for(Rank i=0;i<value.size()-1;i++){
      out << value[i] << " ";
    }
    if(value.size()!=0){
      out << value[value.size()-1];
    }
    return out;
  }

  template<Device device, class Base>
  std::ostream& operator<<(std::ostream& out, const Node<device, Base>& value){
    return out << "[dims(" << value.dims << ") data(" << value.data << ")]";
  }
}
using node::Node;

namespace tensor{
  template<Device device=Device::CPU, class Base=double>
  class Tensor{
  public:
    std::vector<Legs> legs;
    Node<device, Base> node;

    Tensor() = delete;
    ~Tensor() = default;
    Tensor(Tensor<device, Base>&& other) = default;
    Tensor(const Tensor<device, Base>& other) = default;
    Tensor<device, Base>& operator=(Tensor<device, Base>&& other) = default;
    Tensor<device, Base>& operator=(const Tensor<device, Base>& other) = default;
    template<class T1=std::vector<Size>, class T2=std::vector<Legs>>
    Tensor(T1&& _dims, T2&& _legs) : node(std::forward<T1>(_dims)), legs(std::forward<T2>(_legs)){}
    void generate_test(){
      node.generate_test();
    }
  };

  std::ostream& operator<<(std::ostream& out, const std::vector<Legs>& value){
    for(Rank i=0;i<value.size()-1;i++){
      out << value[i] << " ";
    }
    if(value.size()!=0){
      out << value[value.size()-1];
    }
    return out;
  }

  template<Device device, class Base>
  std::ostream& operator<<(std::ostream& out, const Tensor<device, Base>& value){
    return out << "[legs(" << value.legs << ") node(" << value.node << ")]";
  }
}
using tensor::Tensor;

int main(){
  Tensor<> t1({2,3},{Up, Down});
  std::cout << t1 << "\n";
  t1.generate_test();
  std::cout << t1 << "\n";
  t1.generate_test();
  return 0;
}
