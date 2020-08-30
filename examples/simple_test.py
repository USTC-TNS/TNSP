#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def run_test(function):
    print("#", function.__name__)
    function()
    print()


def test_create_tensor():
    print(TAT.Tensor.ZNo(["Left", "Right"], [3, 4]).test())
    print(TAT.Tensor.ZNo(["Left", "Right"], [0, 3]))
    print(TAT.Tensor.DNo([], []).set(lambda: 10))
    print(TAT.Tensor.DNo([], []).set(lambda: 10)[{}])
    print(TAT.Tensor.ZNo(["Left", "Right"], [3, 4]).test()[{"Right": 2, "Left": 1}])


def test_create_symmetry_tensor():
    print(TAT.Tensor.DZ2(["Left", "Right", "Up"], [{1: 3, 0: 1}, {1: 1, 0: 2}, {1: 2, 0: 3}]).zero())
    print(TAT.Tensor.DU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2))
    print(TAT.Tensor.DU1(["Left", "Right", "Up"], [{}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).zero())
    print(TAT.Tensor.DU1([], []).set(lambda: 123))


def test_create_fermi_symmetry_tensor():
    print(TAT.Tensor.DFermi(["Left", "Right", "Up"], [{0: 1, 1: 2}, {-1: 1, -2: 3, 0: 2}, {0: 3, 1: 1}], True).test(2))
    print(TAT.Tensor.DFermiU1(["Left", "Right", "Up"], [{(0, 0): 1, (1, 1): 2}, {(-1, -1): 1, (-2, 0): 3, (0, 0): 2}, {(0, 0): 3, (1, -1): 1}], True).test(2))
    print(TAT.Tensor.DFermiU1(["Left", "Right", "Up"], [{(0, 0): 1, (1, 1): 2}, {(-1, -1): 1, (-2, 0): 3, (0, 0): 2}, {(0, 0): 3, (1, -1): 1}], True).test(2).block({"Left": (1, 1), "Up": (1, -1), "Right": (-2, 0)}))
    print(TAT.Tensor.DFermiU1(1234)[{}])
    print(TAT.Tensor.DFermiU1(["Left", "Right", "Up"], [{(0, 0): 1, (1, 1): 2}, {(-1, -1): 1, (-2, 0): 3, (0, 0): 2}, {(0, 0): 3, (1, -1): 1}], True).test(2)[{"Left": ((1, 1), 1), "Up": ((1, -1), 0), "Right": ((-2, 0), 0)}])


def test_type_conversion():
    print(TAT.Tensor.DU1(123))
    print(TAT.Tensor.DU1(123).value())
    print(TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test(2).to_double_real())
    print(TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test(2).to_double_complex())
    print(TAT.Tensor.ZNo(["Left", "Right"], [3, 4]).test(2).to_double_real())
    print(TAT.Tensor.DU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2).to_double_complex())


def test_norm():
    t = TAT.Tensor.DU1("Left Right Up".split(" "), [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2).to_double_complex()
    print(t.norm_max())
    print(t.norm_num())
    print(t.norm_1())
    print(t.norm_2())


def test_scalar():
    t = TAT.Tensor.DZ2(["Left", "Right", "Phy"], [{0: 2, 1: 2}, {0: 2, 1: 2}, {0: 2, 1: 2}])
    t.test()
    print(t + 1.0)
    print(1.0 / t)

    a = TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test()
    b = TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test(0, 0.1)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a + b.transpose(["Right", "Left"]))


def test_io():
    a = TAT.Tensor.DNo(["Left", "Right", "Up"], [2, 3, 4]).test()
    ss = a.save()
    b = TAT.Tensor.DNo().load(ss)
    print(a)
    print(b)

    c = TAT.Tensor.DU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2)
    ss = c.save()
    d = TAT.Tensor.DU1().load(ss)
    print(c)
    print(d)

    g = TAT.Tensor.ZU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2)
    h = TAT.Tensor.ZU1().load(g.save())
    print(g)
    print(h)

    i = TAT.Tensor.ZFermi(["Left", "Right", "Up"], [{-2: 3, 0: 1, -1: 2}, {0: 2, 1: 3}, {0: 3, 1: 1}], True).test(2)
    j = TAT.Tensor.ZFermi().load(i.save())
    print(i)
    print(j)


def test_edge_rename():
    t1 = TAT.Tensor.DZ2(["Left", "Right", "Phy"], [{0: 1, 1: 2}, {0: 3, 1: 4}, {0: 5, 1: 6}])
    t2 = t1.edge_rename({"Left": "Up"})
    t1.test()
    print(t1)
    print(t2)


def test_transpose():
    a = TAT.Tensor.DNo(["Left", "Right"], [2, 3]).test()
    print(a)
    print(a.transpose(["Right", "Left"]))

    b = TAT.Tensor.DNo(["Left", "Right", "Up"], [2, 3, 4]).test()
    print(b)
    print(b.transpose(["Right", "Up", "Left"]))

    c = TAT.Tensor.ZU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(1)
    print(c)
    ct = c.transpose(["Right", "Up", "Left"])
    print(ct)
    print(c[{"Left": (-1, 0), "Right": (1, 2), "Up": (0, 0)}])
    print(ct[{"Left": (-1, 0), "Right": (1, 2), "Up": (0, 0)}])

    d = TAT.Tensor.DFermi(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}], True) .test(1)
    print(d)
    dt = d.transpose(["Right", "Up", "Left"])
    print(dt)

    e = TAT.Tensor.DNo(["Down", "Up", "Left", "Right"], [2, 3, 4, 5]).test(1)
    print(e)
    et = e.transpose(["Left", "Down", "Right", "Up"])
    print(et)

    f = TAT.Tensor.DNo(["l1", "l2", "l3"], [2, 3, 4]).test()
    print(f)
    print(f.transpose(["l1", "l2", "l3"]))
    print(f.transpose(["l1", "l3", "l2"]))
    print(f.transpose(["l2", "l1", "l3"]))
    print(f.transpose(["l2", "l3", "l1"]))
    print(f.transpose(["l3", "l1", "l2"]))
    print(f.transpose(["l3", "l2", "l1"]))


def test_split_and_merge():
    class initializer():
        def __init__(self):
            self.i = -1

        def __call__(self):
            self.i += 1
            return self.i
    a = TAT.Tensor.DNo(["Left", "Right"], [2, 3]).set(initializer())
    b = a.merge_edge({"Merged": ["Left", "Right"]})
    c = a.merge_edge({"Merged": ["Right", "Left"]})
    # d = c.split_edge({"Merged": [("1", 3), ("2", 2)]});
    print(a)
    print(b)
    print(c)
    # print(d)


if __name__ == "__main__":
    run_test(test_create_tensor)
    run_test(test_create_symmetry_tensor)
    run_test(test_create_fermi_symmetry_tensor)
    run_test(test_type_conversion)
    run_test(test_norm)
    run_test(test_scalar)
    run_test(test_io)
    run_test(test_edge_rename)
    run_test(test_transpose)
    run_test(test_split_and_merge)
