/* example/lazy.cpp
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
//#define TAT_USE_MKL

#include <TAT.hpp>

int main() {
    auto a = std::make_shared<TAT::Lazy<int>>(1);
    auto b = TAT::Lazy<int>::make_lazy([](int x){return x+1;}, a);
    auto c = TAT::Lazy<int>::make_lazy([](int x, int y){return x+y;}, a, b);
    std::cout << a->calc() << "+1=" << b->calc() << ", " << a->calc() << "+" << b->calc() << "=" << c->calc() << std::endl;
    a->reset()->set(10);
    std::cout << a->calc() << "+" << b->calc() << "=" << c->calc() << std::endl;
    return 0;
}