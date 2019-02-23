#include "tensor.hpp"

using T = Node::Tensor<Node::Device::CPU>;

int main()
{
  int stream=0;
  { //test shuffle
    {
      T a(2, {2,3}, {Down, Up});
      a.set_test_data();
      T b;
      a.shuffle_to(b, {Up,Down}, stream);
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
    }
    {
      T a(4, {2,3,4,5}, {Down, Up, Left, Right});
      a.set_test_data();
      T b;
      a.shuffle_to(b, {Left,Down,Right,Up}, stream);
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
      c.contract_from(a,b,{Up},{Up},stream,{},{{Down, Down1}});
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
      c.contract_from(a,b,{Up, Right},{Up,Down},stream,{},{{Left,Left3}});
      std::cout << a << std::endl;
      std::cout << a.content << std::endl;
      std::cout << b << std::endl;
      std::cout << b.content << std::endl;
      std::cout << c << std::endl;
      std::cout << c.content << std::endl;
    }
  }
}
