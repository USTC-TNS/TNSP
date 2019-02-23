#include "tensor.hpp"

using T = Node::Tensor<Node::Device::CPU>;

int main()
{
  { //test shuffle
    int stream=0;
    T a(4, {2,3,4,5}, {Down, Up, Left, Right});
    T b;
    a.shuffle_to(b, {Left,Down,Right,Up}, stream);
    std::cout << a << std::endl << b << std::endl;
  }
  { // test contract
    int stream=0;
    T a(5, {2,3,4,5,6}, {Down, Up, Left, Right,Phy});
    T b(3, {5,3,7}, {Down, Up, Left});
    T c;
    c.contract_from(a,b,{Up, Right},{Up,Down},stream,{},{{Left,Left3}});
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
  }
}
