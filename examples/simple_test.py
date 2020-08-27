#!/usr/bin/env python
# Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import TAT

def test_create_tensor():
    print("# test_create_tensor")
    print(TAT.Tensor.ZNo(["Left", "Right"], [3, 4]).test())
    print(TAT.Tensor.ZNo(["Left", "Right"], [0, 3]))
    a = TAT.Tensor.ZNo([], []).set(lambda :10)
    print(a)
    print(a[{}])
    a[{}] = 20
    print(a)
    print(a.block())
    a.block()[0] = 30
    print(a)
    print("")

def main():
    test_create_tensor()

if __name__ == "__main__":
    main()
