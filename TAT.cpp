#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <cassert>
#include <type_traits>
#include <functional>
#include <set>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ <<std::endl;
#define ENABLE_IF(...) class = typename std::enable_if<__VA_ARGS__::value>::type
#define TAT_USE_CPU
#define TAT_TEST

#ifdef TAT_USE_CPU
extern "C"
{
#include <cblas.h>
}
#include <hptt.h>
#endif

namespace TAT{

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
}
using data::Data;

namespace node{
  template<Device device, class Base>
  class Node;
}
using node::Node;

namespace tensor{
  template<Device device=Device::CPU, class Base=double>
  class Tensor;
}
using tensor::Tensor;

namespace data{
#ifdef TAT_USE_CPU
  namespace transpose {}

  namespace contract {
    template<class Base>
    void gemm(Base* data,
              const Base* data1,
              const Base* data2,
              Size m,
              Size n,
              Size k);

    template<>
    void gemm<double>(double* data,
                      const double* data1,
                      const double* data2,
                      Size m,
                      Size n,
                      Size k)
    {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m, n, k,
                  1, const_cast<double*>(data1), k, const_cast<double*>(data2), n,
                  0, data, n);
    }

    template<>
    void gemm<float>(float* data,
                     const float* data1,
                     const float* data2,
                     Size m,
                     Size n,
                     Size k)
    {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m, n, k,
                  1, const_cast<float*>(data1), k, const_cast<float*>(data2), n,
                  0, data, n);
    }
  }
  template<class Base>
  class Data<Device::CPU, Base>{
    Data() = default;
    friend class Node<Device::CPU, Base>;
    template<Device device2, class Base2, class>
    friend class Data;
  public:
    static Data<Device::CPU, Base> get_empty_data(){
      return Data();
    }

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

    template<class Base2>
    Data<Device::CPU, Base2> to() const {
      Data<Device::CPU, Base2> res;
      res.size = size;
      res.base = std::unique_ptr<Base2[]>(new Base2[size]);
      for(Size i=0;i<size;i++){
        res.base[i] = base[i];
      }
      return res;
    }

    Data<Device::CPU, Base> transpose(const std::vector<Size>& dims,
                                      const std::vector<Rank>& plan,
                                      const std::vector<Size>& new_dims) const {
      Data<Device::CPU, Base> res(size);
      std::vector<int> int_plan(plan.begin(), plan.end());
      std::vector<int> int_dims(dims.begin(), dims.end());
      hptt::create_plan(int_plan.data(), int_plan.size(),
                        1, base.get(), int_dims.data(), NULL,
                        0, res.base.get(), NULL,
                        hptt::ESTIMATE, 1, NULL, 1)->execute();
      return res;
    }

    static Data<Device::CPU, Base> contract(const Data<Device::CPU, Base>& data1,
                                            const Data<Device::CPU, Base>& data2,
                                            const std::vector<Size>& dims1,
                                            const std::vector<Size>& dims2,
                                            const std::vector<Rank>& plan1,
                                            const std::vector<Rank>& plan2,
                                            const std::vector<Size>& new_dims1,
                                            const std::vector<Size>& new_dims2,
                                            const Size& m, const Size& k, const Size&n){
      Data<Device::CPU, Base> a = data1.transpose(dims1, plan1, new_dims1);
      Data<Device::CPU, Base> b = data1.transpose(dims2, plan2, new_dims2);
      Data<Device::CPU, Base> res(m*n);
      contract::gemm<Base>(res.base.get(), a.base.get(), b.base.get(), m, n, k);
      return res;
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
    Data<Device::CPU, Base> operator+(const Data<Device::CPU, Base>& a){
      return Data<Device::CPU, Base>(a);
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

namespace node{
  namespace transpose{
    void plan(std::vector<Size>& new_dims, const std::vector<Size>& dims, const std::vector<Rank>& plan){
      const Rank& rank = dims.size();
      for(Rank i=0;i<rank;i++){
        new_dims.push_back(dims[plan[i]]);
      }
    }
  }

  namespace contract{
    void plan(std::vector<Size>& dims, Size& m, Size& k, Size& n, const::std::vector<Size>& dims1, const::std::vector<Size>& dims2, const Rank& contract_num){
      Rank i, tmp=dims1.size()-contract_num;
      for(i=0;i<tmp;i++){
        m *= dims1[i];
        dims.push_back(dims1[i]);
      }
      for(i=0;i<contract_num;i++){
        k *= dims1[i+tmp];
        assert(dims1[i+tmp]==dims2[i]);
      }
      for(;i<dims2.size();i++){
        n *= dims2[i];
        dims.push_back(dims2[i]);
      }
    }
  }

  template<Device device, class Base>
  class Node{
    Node() = default;
    friend class Tensor<device, Base>;
    template<Device device2, class Base2>
    friend class Node;
  public:
    static Node<device, Base> get_empty_node(){
      return Node();
    }

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

    template<class Base2>
    Node<device, Base2> to() const {
      Node<device, Base2> res;
      res.dims = dims;
      res.data = data.template to<Base2>();
      return res;
    }

    Node<device, Base> transpose(const std::vector<Rank>& plan) const {
      Node<device, Base> res;
      transpose::plan(res.dims, dims, plan);
      assert(get_size(res.dims)==data.size);
      res.data = data.transpose(dims, plan, res.dims);
      return res;
    }

    static Node<device, Base> contract(const Node<device, Base>& node1,
                                       const Node<device, Base>& node2,
                                       const std::vector<Rank>& plan1,
                                       const std::vector<Rank>& plan2,
                                       const Rank& contract_num){
      Node<device, Base> res;
      Size m=1, k=1, n=1;
      std::vector<Size> dims1, dims2;
      transpose::plan(dims1, node1.dims, plan1);
      transpose::plan(dims2, node2.dims, plan2);
      contract::plan(res.dims, m, k, n, dims1, dims2, contract_num);
      res.data = Data<device, Base>::contract(node1.data, node2.data, node1.dims, node2.dims, plan1, plan2, dims1, dims2, m, k, n);
      return res;
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
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = a.data * b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator*(B b, const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
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
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = a.data / b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator/(B b, const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = b / a.data;
      return res;
    }

    template<Device device, class Base>
    Node<device, Base> operator+(const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = + a.data;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base>& operator+=(Node<device, Base>& a, B b){
      a.data += b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator+(const Node<device, Base>& a, B b){
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = a.data + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator+(B b, const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = b + a.data;
      return res;
    }

    template<Device device, class Base>
    Node<device, Base> operator-(const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
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
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = a.data - b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Node<device, Base> operator-(B b, const Node<device, Base>& a){
      auto res = Node<device, Base>::get_empty_node();
      res.dims = a.dims;
      res.data = b - a.data;
      return res;
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
    Node<device, Base1>& operator+=(Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      a.data += b.data;
      return a;
    }

    template<Device device, class Base1, class Base2>
    Node<device, decltype(Base1()+Base2())> operator+(const Node<device, Base1>& a, const Node<device, Base2>& b){
      assert(a.dims==b.dims);
      auto res = Node<device, decltype(Base1()+Base2())>::get_empty_node();
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
      auto res = Node<device, decltype(Base1()-Base2())>::get_empty_node();
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

namespace tensor{
  namespace transpose{
    void plan(std::vector<Rank>& plan, const std::vector<Legs>& new_legs, const std::vector<Legs>& legs)
    {
      const Rank& rank = legs.size();
      for(Rank i=0;i<rank;i++){
        for(Rank j=0;j<rank;j++){
          if(new_legs[i]==legs[j]){
            plan.push_back(j);
            break;
          }
        }
      }
    }
  }

  namespace contract{
    void plan(std::vector<Legs>& legs,
              std::vector<Legs>& new_legs1,
              std::vector<Legs>& new_legs2,
              const std::vector<Legs>& total_legs1,
              const std::vector<Legs>& total_legs2,
              const std::vector<Legs>& legs1,
              const std::vector<Legs>& legs2,
              const std::map<Legs, Legs>& map1,
              const std::map<Legs, Legs>& map2)
    {
      for(auto i : total_legs1){
        auto pos = std::find(legs1.begin(), legs1.end(), i);
        if(pos == legs1.end()){
          new_legs1.push_back(i);
          try{
            legs.push_back(map1.at(i));
          }catch(const std::out_of_range& e){
            legs.push_back(i);
          }
        }
      }
      new_legs1.insert(new_legs1.end(), legs1.begin(), legs1.end());

      new_legs2.insert(new_legs2.end(), legs2.begin(), legs2.end());
      for(auto i : total_legs2){
        auto pos = std::find(legs2.begin(), legs2.end(), i);
        if(pos == legs2.end()){
          new_legs2.push_back(i);
          try{
            legs.push_back(map2.at(i));
          }catch(const std::out_of_range& e){
            legs.push_back(i);
          }
        }
      }
    }
  }

  template<Device device, class Base>
  class Tensor{
    Tensor() = default;
    template<Device device2, class Base2>
    friend class Tensor;
  public:
    static Tensor<device, Base> get_empty_tensor(){
      return Tensor<device, Base>();
    }

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
      assert(std::set<Legs>(legs.begin(), legs.end()).size()==legs.size());
    }

    void set_test(){
      node.set_test();
    }
    void set_zero(){
      node.set_zero();
    }

    template<class Base2>
    Tensor<device, Base2> to() const {
      Tensor<device, Base2> res;
      res.legs = legs;
      res.node = node.template to<Base2>();
      return res;
    }

    template<class T=std::vector<Legs>>
    Tensor<device, Base> transpose(T&& new_legs) const {
      Tensor<device, Base> res;
      res.legs = new_legs;
      std::vector<Rank> plan;
      transpose::plan(plan, res.legs, legs);
      assert(new_legs.size()==legs.size());
      assert(plan.size()==legs.size());
      res.node = node.transpose(plan);
      return res;
    }

    static Tensor<device, Base> contract(const Tensor<device, Base>& tensor1,
                                         const Tensor<device, Base>& tensor2,
                                         const std::vector<Legs> legs1,
                                         const std::vector<Legs> legs2,
                                         const std::map<Legs, Legs>& map1 = {},
                                         const std::map<Legs, Legs>& map2 = {}){
      Tensor<device, Base> res;
      std::vector<Legs> new_legs1, new_legs2;
      std::vector<Rank> plan1, plan2;
      Rank contract_num = legs1.size();
      assert(legs1.size()==legs2.size());
      contract::plan(res.legs, new_legs1, new_legs2, tensor1.legs, tensor2.legs, legs1, legs2, map1, map2);
      transpose::plan(plan1, new_legs1, tensor1.legs);
      transpose::plan(plan2, new_legs2, tensor1.legs);
      res.node = Node<device, Base>::contract(tensor1.node, tensor2.node, plan1, plan2, contract_num);
      return res;
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
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = a.node * b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator*(B b, const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
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
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = a.node / b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator/(B b, const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = b / a.node;
      return res;
    }

    template<Device device, class Base>
    Tensor<device, Base> operator+(const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = + a.node;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base>& operator+=(Tensor<device, Base>& a, B b){
      a.node += b;
      return a;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator+(const Tensor<device, Base>& a, B b){
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = a.node + b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator+(B b, const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = b + a.node;
      return res;
    }

    template<Device device, class Base>
    Tensor<device, Base> operator-(const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
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
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = a.node - b;
      return res;
    }

    template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
    Tensor<device, Base> operator-(B b, const Tensor<device, Base>& a){
      auto res = Tensor<device, Base>::get_empty_tensor();
      res.legs = a.legs;
      res.node = b - a.node;
      return res;
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
    Tensor<device, Base1>& operator+=(Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      a.node += b.node;
      return a;
    }

    template<Device device, class Base1, class Base2>
    Tensor<device, decltype(Base1()+Base2())> operator+(const Tensor<device, Base1>& a, const Tensor<device, Base2>& b){
      assert(a.legs==b.legs);
      auto res = Tensor<device, decltype(Base1()+Base2())>::get_empty_tensor();
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
      auto res = Tensor<device, decltype(Base1()-Base2())>::get_empty_tensor();
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
} // namespace TAT

#ifdef TAT_TEST
using namespace TAT;
int main(){
  std::cout << "scalar\n";
  { // scalar
    {
      Tensor<> t1({2,3},{Up, Down});
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      t1 += 1.2;
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      t1 -= 1.2;
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      t1 *= 1.2;
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      t1 /= 1.2;
      std::cout << t1 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      Tensor<> t2({2,3},{Up, Down});
      t1.set_test();
      t2.set_test();
      t1 += t2;
      std::cout << t1*2.3 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      Tensor<> t2({2,3},{Up, Down});
      t1.set_zero();
      t2.set_test();
      t1 -= t2;
      std::cout << 1-t1/3.4 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      Tensor<> t2({2,3},{Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << 1+3/(t1+1)+t2 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      Tensor<> t2({2,3},{Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << +(t1-1.2)-t2 << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      std::cout << 3+1.2/(t1*1.2) << "\n";
    }
    {
      Tensor<> t1({2,3},{Up, Down});
      t1.set_test();
      std::cout << -(2.4*(t1/1.2)) << "\n";
    }
  } // scalar
  std::cout << "transpose\n";
  { // transpose
    {
      Tensor<> t1({2,3},{Left,Right});
      t1.set_test();
      auto t2 = t1.transpose({Right,Left});
      std::cout << t1 << "\n" << t2 << "\n";
    }
    {
      Tensor<> t1({2,3,4,5},{Down,Up,Left,Right});
      t1.set_test();
      auto t2 = t1.transpose({Left,Down,Right,Up});
      std::cout << t1 << "\n" << t2 << "\n";
    }
  } // transpose
  std::cout << "to\n";
  { // to
    {
      Tensor<> t1({2,3},{Left,Right});
      t1.set_test();
      Tensor<Device::CPU, int> t2 = t1.to<int>();
      std::cout << t1 << "\n" << t2 << "\n";
    }
  } // to
  std::cout << "contract\n";
  { // contract
    Tensor<> t1({2,3}, {Down, Up});
    Tensor<> t2({2,3}, {Down, Up});
    t1.set_test();
    t2.set_test();
    std::cout << t1 << "\n" << t2 << "\n" << Tensor<>::contract(t1, t2, {Up}, {Up}, {}, {{Down, Down1}}) << "\n";
  } // contract
}
#endif // TAT_TEST
