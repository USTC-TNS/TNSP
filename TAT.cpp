#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <cassert>
#include <type_traits>
#include <functional>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ <<std::endl;
#define ENABLE_IF(...) typename = typename std::enable_if<__VA_ARGS__::value>::type
#define TAT_USE_CPU

enum class Device {CPU, CUDA, DCU, SW};

namespace legs{
  enum class Legs
    {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
     CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4),
     CreateLeg(5), CreateLeg(6), CreateLeg(7), CreateLeg(8), CreateLeg(9)
#undef CreateLeg
    };

  inline namespace io{}
  namespace io{
#define IncEnum(p) {Legs::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
    static const std::map<Legs, std::string> legs_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
                                                         IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)};
#undef IncGroup
#undef IncEnum

    std::ostream& operator<<(std::ostream& out, const Legs& value){
      try{
        return out << legs_str.at(value);
      }catch(const std::out_of_range& e){
        return out;
      }
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
  template<Device device, class Base, ENABLE_IF(std::is_scalar<Base>)>
  class Data;

#ifdef TAT_USE_CPU
  template<class Base>
  class Data<Device::CPU, Base>{
    Data() = default;
  public:
    Size size;
    std::unique_ptr<Base[]> base;

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

    void set_test(){
      for(Size i=0;i<size;i++){
        base[i] = i;
      }
    }
    void set_zero(){
      for(Size i=0;i<size;i++){
        base[i] = 0;
      }
    }
  };

  inline namespace scalar{}
  namespace scalar{
    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base>& operator*=(Data<Device::CPU, Base>& a, B b){
      Base bb = b;
      for(Size i=0;i<a.size;i++){
        a.base[i] *= bb;
      }
      return a;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator*(const Data<Device::CPU, Base>& a, B b){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] * bb;
      }
      return res;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator*(B b, const Data<Device::CPU, Base>& a){
      return a * b;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base>& operator/=(Data<Device::CPU, Base>& a, B b){
      Base bb = b;
      for(Size i=0;i<a.size;i++){
        a.base[i] /= bb;
      }
      return a;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator/(const Data<Device::CPU, Base>& a, B b){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] / bb;
      }
      return res;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator/(B b, const Data<Device::CPU, Base>& a){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = bb / a.base[i];
      }
      return res;
    }

    template<class Base>
    Data<Device::CPU, Base>& operator+(Data<Device::CPU, Base>& a){
      return a;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base>& operator+=(Data<Device::CPU, Base>& a, B b){
      Base bb = b;
      for(Size i=0;i<a.size;i++){
        a.base[i] += bb;
      }
      return a;
    }


    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator+(const Data<Device::CPU, Base>& a, B b){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] + bb;
      }
      return res;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator+(B b, const Data<Device::CPU, Base>& a){
      return a + b;
    }

    template<class Base>
    Data<Device::CPU, Base> operator-(const Data<Device::CPU, Base>& a){
      Data<Device::CPU, Base> res(a.size);
      for(Size i=0;i<res.size;i++){
        res.base[i] = - a.base[i];
      }
      return res;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base>& operator-=(Data<Device::CPU, Base>& a, B b){
      Base bb = b;
      for(Size i=0;i<a.size;i++){
        a.base[i] -= bb;
      }
      return a;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator-(const Data<Device::CPU, Base>& a, B b){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] - bb;
      }
      return res;
    }

    template<class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Data<Device::CPU, Base> operator-(B b, const Data<Device::CPU, Base>& a){
      Data<Device::CPU, Base> res(a.size);
      Base bb = b;
      for(Size i=0;i<res.size;i++){
        res.base[i] = bb - a.base[i];
      }
      return res;
    }

    template<class Base1, class Base2>
    Data<Device::CPU, Base1>& operator+=(Data<Device::CPU, Base1>& a, const Data<Device::CPU, Base2>& b){
      assert(a.size==b.size);
      for(Size i=0;i<a.size;i++){
        a.base[i] += b.base[i];
      }
      return a;
    }

    template<class Base1, class Base2>
    Data<Device::CPU, decltype(Base1()+Base2())> operator+(const Data<Device::CPU, Base1>& a, const Data<Device::CPU, Base2>& b){
      assert(a.size==b.size);
      Data<Device::CPU, decltype(Base1()+Base2())> res(a.size);
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] + b.base[i];
      }
      return res;
    }

    template<class Base1, class Base2>
    Data<Device::CPU, Base1>& operator-=(Data<Device::CPU, Base1>& a, const Data<Device::CPU, Base2>& b){
      assert(a.size==b.size);
      for(Size i=0;i<a.size;i++){
        a.base[i] -= b.base[i];
      }
      return a;
    }

    template<class Base1, class Base2>
    Data<Device::CPU, decltype(Base1()-Base2())> operator-(const Data<Device::CPU, Base1>& a, const Data<Device::CPU, Base2>& b){
      assert(a.size==b.size);
      Data<Device::CPU, decltype(Base1()-Base2())> res(a.size);
      for(Size i=0;i<res.size;i++){
        res.base[i] = a.base[i] - b.base[i];
      }
      return res;
    }
  }

  inline namespace io{}
  namespace io{
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
  } // namespace io
#endif // TAT_USE_CPU
} // namespace data
using data::Data;

namespace node{
  template<Device device, class Base>
  class Node{
    Node() = default;
  public:
    std::vector<Size> dims;
    Data<device, Base> data;

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

    void set_test(){
      data.set_test();
    }
    void set_zero(){
      data.set_zero();
    }
  };

  inline namespace scalar{}
  namespace scalar{
    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator*=(Node<device, Base>& a, B b){
      a.data *= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator*(const Node<device, Base>& a, B b){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = a.data * b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator*(B b, const Node<device, Base>& a){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = b * a.data;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator/=(Node<device, Base>& a, B b){
      a.data /= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator/(const Node<device, Base>& a, B b){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = a.data / b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator/(B b, const Node<device, Base>& a){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = b / a.data;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator+(Node<device, Base>& a){
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator+=(Node<device, Base>& a, B b){
      a.data += b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator+(const Node<device, Base>& a, B b){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = a.data + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator+(B b, const Node<device, Base>& a){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = a.data + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator-(const Node<device, Base>& a){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = - a.data;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator-=(Node<device, Base>& a, B b){
      a.data -= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator-(const Node<device, Base>& a, B b){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = a.data - b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator-(B b, const Node<device, Base>& a){
      Node<device, Base> res;
      res.dims = a.dims;
      res.data = b - a.data;
      return res;
    }

    template<Device device, class Base1, class Base2>
    Node<device, Base1>& operator+=(Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      a.data += b.data;
      return a;
    }

    bool operator==(const std::vector<Size>& a, const std::vector<Size>& b){
      if(a.size()!=b.size()){
        return false;
      }
      for(Rank i=0;i<a.size();i++){
        if(a[i]!=b[i]){
          return false;
        }
      }
      return true;
    }

    template<Device device, class Base1, class Base2>
    Node<device, decltype(Base1()+Base2())> operator+(const Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      Node<device, decltype(Base1()+Base2())> res;
      res.dims = a.dims;
      res.data = a.data + b.data;
      return res;
    }

    template<Device device, class Base1, class Base2>
    Node<device, Base1>& operator-=(Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      a.data -= b.data;
      return a;
    }

    template<Device device, class Base1, class Base2>
    Node<device, decltype(Base1()-Base2())> operator-(const Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      Node<device, decltype(Base1()+Base2())> res;
      res.dims = a.dims;
      res.data = a.data - b.data;
      return res;
    }
  }

  inline namespace io{}
  namespace io{
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
}
using node::Node;

namespace tensor{
  template<Device device=Device::CPU, class Base=double>
  class Tensor{
    Tensor() = default;
  public:
    std::vector<Legs> legs;
    Node<device, Base> node;

    ~Tensor() = default;
    Tensor(Tensor<device, Base>&& other) = default;
    Tensor(const Tensor<device, Base>& other) = default;
    Tensor<device, Base>& operator=(Tensor<device, Base>&& other) = default;
    Tensor<device, Base>& operator=(const Tensor<device, Base>& other) = default;
    template<class T1=std::vector<Size>, class T2=std::vector<Legs>>
    Tensor(T1&& _dims, T2&& _legs) : legs(std::forward<T2>(_legs)), node(std::forward<T1>(_dims)) { 
      assert(legs.size()==node.dims.size());
    }

    void set_test(){
      node.set_test();
    }
    void set_zero(){
      node.set_zero();
    }
  };

  inline namespace scalar{}
  namespace scalar{
    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator*=(Tensor<device, Base>& a, B b){
      a.node *= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator*(const Tensor<device, Base>& a, B b){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = a.node * b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator*(B b, const Tensor<device, Base>& a){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = b * a.node;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator/=(Tensor<device, Base>& a, B b){
      a.node /= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator/(const Tensor<device, Base>& a, B b){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = a.node / b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator/(B b, const Tensor<device, Base>& a){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = b / a.node;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator+(Tensor<device, Base>& a){
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator+=(Tensor<device, Base>& a, B b){
      a.node += b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator+(const Tensor<device, Base>& a, B b){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = a.node + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator+(B b, const Tensor<device, Base>& a){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = a.node + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator-(const Tensor<device, Base>& a){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = - a.node;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator-=(Tensor<device, Base>& a, B b){
      a.node -= b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator-(const Tensor<device, Base>& a, B b){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = a.node - b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator-(B b, const Tensor<device, Base>& a){
      Tensor<device, Base> res;
      res.legs = a.legs;
      res.node = b - a.node;
      return res;
    }

    template<Device device, class Base1, class Base2>
    Tensor<device, Base1>& operator+=(Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      a.node += b.node;
      return a;
    }

    bool operator==(const std::vector<Legs>& a, const std::vector<Legs>& b){
      if(a.size()!=b.size()){
        return false;
      }
      for(Rank i=0;i<a.size();i++){
        if(a[i]!=b[i]){
          return false;
        }
      }
      return true;
    }

    template<Device device, class Base1, class Base2>
    Tensor<device, decltype(Base1()+Base2())> operator+(const Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      Tensor<device, decltype(Base1()+Base2())> res;
      res.legs = a.legs;
      res.node = a.node + b.node;
      return res;
    }

    template<Device device, class Base1, class Base2>
    Tensor<device, Base1>& operator-=(Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      a.node -= b.node;
      return a;
    }

    template<Device device, class Base1, class Base2>
    Tensor<device, decltype(Base1()-Base2())> operator-(const Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      Tensor<device, decltype(Base1()-Base2())> res;
      res.legs = a.legs;
      res.node = a.node - b.node;
      return res;
    }
  }

  inline namespace io{}
  namespace io{
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
}
using tensor::Tensor;

int main(){
  Tensor<> t1({2,3},{Up, Down});
  std::cout << t1 << "\n";
  t1.set_test();
  std::cout << t1 << "\n";
  t1.node.data *= 2;
  std::cout << t1 << "\n";
  t1.node.data = 2/t1.node.data;
  std::cout << t1 << "\n";
  return 0;
}
