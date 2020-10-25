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
    print(
        TAT.Tensor.DFermiU1(["Left", "Right", "Up"], [{
            (0, 0): 1,
            (1, 1): 2
        }, {
            (-1, -1): 1,
            (-2, 0): 3,
            (0, 0): 2
        }, {
            (0, 0): 3,
            (1, -1): 1
        }], True).test(2).block[{
            "Left": (1, 1),
            "Up": (1, -1),
            "Right": (-2, 0)
        }])
    print(TAT.Tensor.DFermiU1(1234)[{}])
    print(
        TAT.Tensor.DFermiU1(["Left", "Right", "Up"], [{
            (0, 0): 1,
            (1, 1): 2
        }, {
            (-1, -1): 1,
            (-2, 0): 3,
            (0, 0): 2
        }, {
            (0, 0): 3,
            (1, -1): 1
        }], True).test(2)[{
            "Left": ((1, 1), 1),
            "Up": ((1, -1), 0),
            "Right": ((-2, 0), 0)
        }])


def test_type_conversion():
    print(TAT.Tensor.DU1(123))
    print(float(TAT.Tensor.DU1(123)))
    print(TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test(2).to_double_real())
    print(TAT.Tensor.DNo(["Left", "Right"], [3, 4]).test(2).to_double_complex())
    print(TAT.Tensor.ZNo(["Left", "Right"], [3, 4]).test(2).to_double_real())
    print(TAT.Tensor.DU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2).to_double_complex())


def test_norm():
    t = TAT.Tensor.DU1("Left Right Up".split(" "), [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2).to_double_complex()
    print(t.norm_max())
    print(t.norm_num())
    print(t.norm_sum())
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
    ss = a.dump()
    b = TAT.Tensor.DNo().load(ss)
    print(a)
    print(b)

    c = TAT.Tensor.DU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2)
    ss = c.dump()
    d = TAT.Tensor.DU1().load(ss)
    print(c)
    print(d)

    g = TAT.Tensor.ZU1(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).test(2)
    h = TAT.Tensor.ZU1().load(g.dump())
    print(g)
    print(h)

    i = TAT.Tensor.ZFermi(["Left", "Right", "Up"], [{-2: 3, 0: 1, -1: 2}, {0: 2, 1: 3}, {0: 3, 1: 1}], True).test(2)
    j = TAT.Tensor.ZFermi().load(i.dump())
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

    d = TAT.Tensor.DFermi(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}], True).test(1)
    print(d)
    dt = d.transpose(["Right", "Up", "Left"])
    print(dt)

    e = TAT.Tensor.DNo(["Down", "Up", "Left", "Right"], [2, 3, 4, 5]).test(1)
    print(e)
    et = e.transpose(["Left", "Down", "Right", "Up"])
    print(et)
    print(e[{"Down": 1, "Up": 1, "Left": 2, "Right": 2}])
    print(et[{"Down": 1, "Up": 1, "Left": 2, "Right": 2}])

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

        def __init__(self, first=-1):
            self.i = first

        def __call__(self):
            self.i += 1
            return self.i

    a = TAT.Tensor.DNo(["Left", "Right"], [2, 3]).set(initializer())
    b = a.merge_edge({"Merged": ["Left", "Right"]})
    c = a.merge_edge({"Merged": ["Right", "Left"]})
    d = c.split_edge({"Merged": [("1", 3), ("2", 2)]})
    print(a)
    print(b)
    print(c)
    print(d)

    e = TAT.Tensor.ZFermi(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).set(initializer(0))
    print(e)
    f = e.merge_edge({"Merged": ["Left", "Up"]})
    print(f)
    g = f.split_edge({"Merged": [("Left", {-1: 3, 0: 1, 1: 2}), ("Up", {-1: 2, 0: 3, 1: 1})]})
    print(g)
    h = g.transpose(["Left", "Right", "Up"])
    print(h)


def test_edge_operator():
    print(TAT.Tensor.DNo(["A", "B"], [8, 8]).test())
    print(TAT.Tensor.DNo(["A", "B"], [8, 8]).test().edge_operator({"A": "C"}, {"C": [("D", 4), ("E", 2)], "B": [("F", 2), ("G", 4)]}, {"D", "F"}, {"I": ["D", "F"], "J": ["G", "E"]}, ["J", "I"]))
    print(TAT.Tensor.DNo(["A", "B", "C"], [2, 3, 4]).test().edge_operator({}, {}, set(), {}, ["B", "C", "A"]))

    a = TAT.Tensor.DU1(["Left", "Right", "Up", "Down"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 4, 1: 2}, {-1: 2, 0: 3, 1: 1}, {-1: 1, 0: 3, 1: 2}]).test(1)
    b = a.edge_rename({"Right": "Right1"}).split_edge({"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]})
    c = b.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_operator({"Right": "Right1"}, {"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]}, set(), {"Left": ["Left", "Down2"]}, ["Down1", "Right1", "Up", "Left"])
    print((total - d).norm_max())

    a = TAT.Tensor.DFermi(["Left", "Right", "Up", "Down"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 4, 1: 2}, {-1: 2, 0: 3, 1: 1}, {-1: 1, 0: 3, 1: 2}]).test(1)
    b = a.edge_rename({"Right": "Right1"}).split_edge({"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]})
    r = b.reverse_edge({"Left"})
    c = r.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_operator({"Right": "Right1"}, {"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]}, {"Left"}, {"Left": ["Left", "Down2"]}, ["Down1", "Right1", "Up", "Left"])
    print((total - d).norm_max())
    print(total)


def test_contract():
    a = TAT.Tensor.DNo(["A", "B"], [2, 2]).test()
    b = TAT.Tensor.DNo(["C", "D"], [2, 2]).test()
    print(a)
    print(b)
    print(a.contract(b, {("A", "C")}))
    print(a.contract(b, {("A", "D")}))
    print(a.contract(b, {("B", "C")}))
    print(a.contract(b, {("B", "D")}))

    print(TAT.Tensor.DNo(["A", "B", "C", "D"], [1, 2, 3, 4]).test().contract(TAT.Tensor.DNo(["E", "F", "G", "H"], [3, 1, 2, 4]).test(), {("B", "G"), ("D", "H")}))

    c = TAT.Tensor.DFermi(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).test()
    d = TAT.Tensor.DFermi(["E", "F", "G", "H"], [{0: 2, 1: 1}, {-2: 1, -1: 1, 0: 2}, {0: 1, -1: 2}, {0: 2, 1: 1, 2: 2}], True).test()
    print(c)
    print(d)
    print(c.contract(d, {("B", "G"), ("D", "H")}))
    print(c.transpose(["A", "C", "B", "D"]).contract(d.transpose(["G", "H", "E", "F"]), {("B", "G"), ("D", "H")}))


def test_svd():
    a = TAT.Tensor.DNo(["A", "B", "C", "D"], [2, 3, 4, 5]).test()
    print(a)
    [u, s, v] = a.svd({"C", "A"}, "E", "F")
    print(u)
    print(v)
    print(s)
    print(v.multiple(s, "F", 'u').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]))

    b = TAT.Tensor.ZNo(["A", "B", "C", "D"], [2, 3, 4, 5]).test()
    print(b)
    [u, s, v] = b.svd({"A", "D"}, "E", "F")
    print(u)
    print(v)
    print(s)
    print(v.contract(v.edge_rename({"F": "F2"}), {("B", "B"), ("C", "C")}).transform(lambda i: i if abs(i) > 1e-5 else 0))
    print(v.multiple(s, "F", 'v').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]))

    c = TAT.Tensor.DU1(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).test()
    [u, s, v] = c.svd({"C", "A"}, "E", "F")
    print(u)
    print(s)
    print(v)
    print(c)
    print(v.multiple(s, "F", 'v').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))
    print(v.contract(u.multiple(s, "E", 'u'), {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))

    a = TAT.Tensor.DNo(["A", "B", "C", "D"], [2, 3, 4, 5]).test()
    print(a)
    [u, s, v] = a.svd({"C", "A"}, "E", "F", 2)
    print(u)
    print(v)
    print(s)
    print(v.multiple(s, "F", 'u').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))

    c = TAT.Tensor.DU1(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).test()
    [u, s, v] = c.svd({"C", "A"}, "E", "F", 7)
    print(u)
    print(s)
    print(v)
    print(c)
    print(v.multiple(s, "F", 'v').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))
    print(v.contract(u.multiple(s, "E", 'u'), {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))


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
    run_test(test_edge_operator)
    run_test(test_contract)
    run_test(test_svd)
