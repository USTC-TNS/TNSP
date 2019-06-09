/* example/test.cpp
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#define TAT_DEFAULT

#include <TAT.hpp>

using TAT::Node;

using namespace TAT::legs_name;

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cout << "scalar\n";
  {
    // scalar
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_zero();
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      t1 += 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      t1 -= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      t1 *= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      t1 /= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      t1 += t2;
      std::cout << t1*2.3 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_zero();
      t2.set_test();
      t1 -= t2;
      std::cout << 1-t1/3.4 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      std::cout << 1+3/(t1+1)+t2 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      std::cout << +(t1-1.2)-t2 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      std::cout << 3+1.2/(t1*1.2) << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      t1.set_test();
      std::cout << -(2.4*(t1/1.2)) << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      std::cout << t1/t2 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      t2+=1;
      t1/=t2;
      std::cout << t1 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      std::cout << t1* t2 << std::endl;
    }
    {
      Node<> t1({Up, Down}, {2, 3});
      Node<> t2({Up, Down}, {2, 3});
      t1.set_test();
      t2.set_test();
      t1 += 1;
      t1 *= t2;
      std::cout << t1 << std::endl;
    }
    {
      //Node<> t1({2},{});
    }
    {
      //Node<> t1({2,3},{Down,Down});
    }
  } // scalar
  std::cout << "transpose\n";
  {
    // transpose
    {
      Node<> t1({Left, Right}, {2, 3});
      t1.set_test();
      auto t2 = t1.transpose({Right, Left});
      std::cout << t1 << std::endl << t2 << std::endl;
    }
    {
      Node<> t1({Down, Up, Left, Right}, {2, 3, 4, 5});
      t1.set_test();
      auto t2 = t1.transpose({Left, Down, Right, Up});
      std::cout << t1 << std::endl << t2 << std::endl;
    }
    {
      Node<> t1({Down, Up, Left, Right}, {2, 3, 4, 5});
      t1.set_test();
      auto t2 = t1.transpose({Left, Down, Right, Phy, Up});
      std::cout << t1 << std::endl << t2 << std::endl;
    }
    {
      //Node<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Down});
    }
    {
      //Node<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Left,Left});
    }
    {
      //Node<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Right});
    }
  } // transpose
  std::cout << "to\n";
  {
    // to
    {
      Node<> t1({Left, Right}, {2, 3});
      t1.set_test();
      Node<TAT::Device::CPU, int> t2 = t1.to<int>();
      std::cout << t1 << std::endl << t2 << std::endl;
    }
  } // to
  std::cout << "contract\n";
  {
    // contract
    {
      Node<> t1({Down, Up}, {2, 3});
      Node<> t2({Down, Up}, {2, 3});
      t1.set_test();
      t2.set_test();
      std::cout << t1 << std::endl << t2 << std::endl << Node<>::contract(t1, t2, {Up}, {Up}, {}, {{Down, Down1}}) << std::endl;
    }
    {
      Node<> t1({Down, Up, Left, Right, Phy}, {2, 3, 4, 5, 6});
      Node<> t2({Down, Up, Left}, {5, 3, 7});
      t1.set_test();
      t2.set_test();
      std::cout << t1 << std::endl << t2 << std::endl << Node<>::contract(t1, t2, {Up, Right}, {Up, Down}, {}, {{Left, Left3}}) << std::endl;
    }
    {
      Node<> t1({Down, Up, Left, Right, Phy}, {2, 3, 4, 5, 6});
      Node<> t2({Down, Up, Left}, {5, 3, 7});
      t1.set_test();
      t2.set_test();
      std::cout << t1 << std::endl << t2 << std::endl << Node<>::contract(t1, t2, {Up, Right, Left3, Right3}, {Up, Down, Left4, Right4}, {{Left2, Right2}}, {{Left, Left3}}) << std::endl;
    }
    {
      //Node<> t1({2,3}, {Down, Up});
      //Node<> t2({2,3}, {Down, Up});
      //Node<>::contract(t1, t2, {Up}, {Left}, {}, {{Down, Down1}});
    }
    {
      //Node<> t1({2,3}, {Down, Up});
      //Node<> t2({2,3}, {Down, Up});
      //Node<>::contract(t1, t2, {Up}, {Down}, {}, {{Up, Down1}});
    }
    {
      //Node<> t1({2,3}, {Down, Up});
      //Node<> t2({2,3}, {Down, Up});
      //Node<>::contract(t1, t2, {Up,Down}, {Up, Up}, {}, {{Up, Down1}});
    }
  } // contract
  std::cout << "multiple\n";
  {
    // multiple
    {
      Node<> t1({Down, Up}, {3, 4});
      Node<> t2({Down}, {4});
      t1.set_test();
      t2.set_test();
      auto t3 = t1.multiple(t2, Up);
      std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
    {
      Node<> t1({Right, Down, Up}, {2, 3, 4});
      Node<> t2({Down}, {3});
      t1.set_test();
      t2.set_test();
      auto t3 = t1.multiple(t2, Down);
      std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
    {
      //Node<> t1({2,3,4}, {Right,Down, Up});
      //Node<> t2({3}, {Down});
      //t1.set_test();
      //t2.set_test();
      //auto t3 = t1.multiple(t2, Up);
      //std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
  } // multiple
  std::cout << "svd\n";
  {
    // svd
    {
      Node<> t1({Left, Right}, {4, 6});
      t1.set_test();
      auto res = t1.svd({Left}, Right, Down, 4);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Node<> t1({Left, Right, Up, Down}, {2, 2, 3, 2});
      t1.set_test();
      auto res = t1.svd({Left, Right}, Right1, Down1);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Node<> t1({Left, Right, Up, Down}, {2, 2, 3, 2});
      t1.set_test();
      auto res = t1.svd({Left, Down}, Right1, Down1, -1);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Node<> t1({Left, Right, Up, Down}, {2, 2, 3, 2});
      t1.set_test();
      auto res = t1.svd({Left, Down}, Right1, Down1, 3);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
      std::ofstream f2;
      f2.open("test_io2.out");
      f2 << res.V;
      f2.close();
    }
  } // svd
  std::cout << "io\n";
  {
    // io
    {
      Node<> t1({Left, Right, Up, Down}, {2, 2, 3, 2});
      t1.set_test();
      std::cout << t1 << std::endl;
      std::ofstream f1;
      f1.open("test_io.out");
      f1 << t1;
      f1.close();
      Node<> t2;
      std::ifstream f2;
      f2.open("test_io.out");
      f2 >> t2;
      f2.close();
      std::cout << t2 << std::endl;
    }
  } // io
  std::cout << "qr\n";
  {
    // qr
    {
      Node<> t1({Left, Right}, {4, 6});
      t1.set_test();
      auto res = t1.qr({Left}, Right, Down);
      std::cout << res.Q << std::endl << res.R << std::endl;
    }
    {
      Node<> t1({Left, Right}, {4, 6});
      t1.set_test();
      auto res = t1.qr({Right}, Up, Down);
      std::cout << res.Q << std::endl << res.R << std::endl;
    }
  } // qr
  return 0;
} // main
