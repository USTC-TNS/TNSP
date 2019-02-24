#include "tensor.hpp"

int main()
{
#ifdef USE_CPU
#ifdef T
#undef T
#endif
#define T Node::Tensor<Node::Device::CPU>
  { //test shuffle
    {
      T a(2, {2,3}, {Down, Up});
      a.set_test_data();
      T b;
      a.shuffle_to(b, {Up,Down});
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
    }
    {
      T a(4, {2,3,4,5}, {Down, Up, Left, Right});
      a.set_test_data();
      T b;
      a.shuffle_to(b, {Left,Down,Right,Up});
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
    }
  }
  { // test contract
    {
      T a(2, {2,3}, {Down, Up});
      T b(2, {2,3}, {Down, Up});
      a.set_test_data();
      b.set_test_data();
      T c;
      c.contract_from(a,b,{Up},{Up},{},{{Down, Down1}});
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
      std::cout << c << std::endl;
      std::cout << c.content << std::endl;
    }
    {
      T a(5, {2,3,4,5,6}, {Down, Up, Left, Right,Phy});
      T b(3, {5,3,7}, {Down, Up, Left});
      a.set_test_data();
      b.set_test_data();
      T c;
      c.contract_from(a,b,{Up, Right},{Up,Down},{},{{Left,Left3}});
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
      std::cout << c << std::endl;
      std::cout << c.content << std::endl;
    }
  }
#endif

#ifdef USE_CUDA
#ifdef T
#undef T
#endif
#define T Node::Tensor<Node::Device::CUDA>
  {
    T a(2, {2, 3}, {Left, Right});
  }
#endif
}
