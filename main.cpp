#include <iostream>
#include "tensor.hpp"

int main()
{
  int stream = 0;
  auto s = {2ul,3ul,4ul,5ul};
  auto l = {Node::Leg::Down, Node::Leg::Up, Node::Leg::Left, Node::Leg::Right};
  auto m = {Node::Leg::Right, Node::Leg::Left, Node::Leg::Down, Node::Leg::Up};
  Node::Tensor<Node::Device::CPU> t(4, s, l), r;
  t.shuffle_to(r, m, stream);
  for(auto i : r.dims)
  {
    std::cout << i << " ";
  }
  std::cout << "\n";
  t.rename_leg({{Node::Leg::Down,Node::Leg::Down1},{Node::Leg::Left,Node::Leg::Right},
    {Node::Leg::Right,Node::Leg::Left},{Node::Leg::Up,Node::Leg::Up1}});
  for(auto i : t.legs)
  {
    std::cout << i << " ";
  }
  std::cout << "\n";
  Node::Tensor<Node::Device::CPU> ok = t;
  ok = r;
}
