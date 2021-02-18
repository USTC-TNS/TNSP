/**
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <iostream>

#include <TAT/TAT.hpp>

int main() {
   TAT::Edge<TAT::FermiSymmetry, TAT::pmr::polymorphic_allocator, false> a;
   TAT::Core<double, TAT::FermiSymmetry, TAT::pmr::polymorphic_allocator> b({{-1, 0, 1}, {-1, 0, 1}});
}
