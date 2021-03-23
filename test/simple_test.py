#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
    print(TAT(complex)(["Left", "Right"], [3, 4]).range())
    print(TAT(complex)(["Left", "Right"], [0, 3]))
    print(TAT(float)([], []).set(lambda: 10))
    print(TAT(float)([], []).set(lambda: 10)[{}])
    print(TAT(complex)(["Left", "Right"], [3, 4]).range()[{"Right": 2, "Left": 1}])


def test_create_symmetry_tensor():
    print(TAT(float, "Z2")(["Left", "Right", "Up"], [{1: 3, 0: 1}, {1: 1, 0: 2}, {1: 2, 0: 3}]).zero())
    print(TAT(float, "U1")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(2))
    print(TAT(float, "U1")(["Left", "Right", "Up"], [{}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).zero())
    print(TAT(float, "U1")([], []).set(lambda: 123))


def test_create_fermi_symmetry_tensor():
    print(TAT(float, "Fermi")(["Left", "Right", "Up"], [{0: 1, 1: 2}, {-1: 1, -2: 3, 0: 2}, {0: 3, 1: 1}], True).range(2))
    print(TAT(float, "FermiU1")(["Left", "Right", "Up"], [{(0, 0): 1, (1, 1): 2}, {(-1, -1): 1, (-2, 0): 3, (0, 0): 2}, {(0, 0): 3, (1, -1): 1}], True).range(2))
    print(
        TAT(float, "FermiU1")(["Left", "Right", "Up"], [{
            (0, 0): 1,
            (1, 1): 2
        }, {
            (-1, -1): 1,
            (-2, 0): 3,
            (0, 0): 2
        }, {
            (0, 0): 3,
            (1, -1): 1
        }], True).range(2).block[{
            "Left": (1, 1),
            "Up": (1, -1),
            "Right": (-2, 0)
        }])
    print(TAT(float, "FermiU1")(1234)[{}])
    print(
        TAT(float, "FermiU1")(["Left", "Right", "Up"], [{
            (0, 0): 1,
            (1, 1): 2
        }, {
            (-1, -1): 1,
            (-2, 0): 3,
            (0, 0): 2
        }, {
            (0, 0): 3,
            (1, -1): 1
        }], True).range(2)[{
            "Left": ((1, 1), 1),
            "Up": ((1, -1), 0),
            "Right": ((-2, 0), 0)
        }])


def test_type_conversion():
    print(TAT(float, "U1")(123))
    print(float(TAT(float, "U1")(123)))
    print(TAT(float)(["Left", "Right"], [3, 4]).range(2).to(float))
    print(TAT(float)(["Left", "Right"], [3, 4]).range(2).to(complex))
    print(TAT(complex)(["Left", "Right"], [3, 4]).range(2).to(float))
    print(TAT(float, "U1")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(2).to(complex))


def test_norm():
    t = TAT(float, "U1")("Left Right Up".split(" "), [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(2).to(complex)
    print(t.norm_max())
    print(t.norm_num())
    print(t.norm_sum())
    print(t.norm_2())


def test_scalar():
    t = TAT(float, "Z2")(["Left", "Right", "Phy"], [{0: 2, 1: 2}, {0: 2, 1: 2}, {0: 2, 1: 2}])
    t.range()
    print(t + 1.0)
    print(1.0 / t)

    a = TAT(float)(["Left", "Right"], [3, 4]).range()
    b = TAT(float)(["Left", "Right"], [3, 4]).range(0, 0.1)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a + b.transpose(["Right", "Left"]))


def test_io():
    a = TAT(float)(["Left", "Right", "Up"], [2, 3, 4]).range()
    ss = a.dump()
    b = TAT(float)().load(ss)
    print(a)
    print(b)

    c = TAT(float, "U1")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(2)
    ss = c.dump()
    d = TAT(float, "U1")().load(ss)
    print(c)
    print(d)

    g = TAT(complex, "U1")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(2)
    h = TAT(complex, "U1")().load(g.dump())
    print(g)
    print(h)

    i = TAT(complex, "Fermi")(["Left", "Right", "Up"], [{-2: 3, 0: 1, -1: 2}, {0: 2, 1: 3}, {0: 3, 1: 1}], True).range(2)
    j = TAT(complex, "Fermi")().load(i.dump())
    print(i)
    print(j)


def test_edge_rename():
    t1 = TAT(float, "Z2")(["Left", "Right", "Phy"], [{0: 1, 1: 2}, {0: 3, 1: 4}, {0: 5, 1: 6}])
    t2 = t1.edge_rename({"Left": "Up"})
    t1.range()
    print(t1)


def test_transpose():
    a = TAT(float)(["Left", "Right"], [2, 3]).range()
    print(a)
    print(a.transpose(["Right", "Left"]))

    b = TAT(float)(["Left", "Right", "Up"], [2, 3, 4]).range()
    print(b)
    print(b.transpose(["Right", "Up", "Left"]))

    c = TAT(complex, "U1")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).range(1)
    print(c)
    ct = c.transpose(["Right", "Up", "Left"])
    print(ct)
    print(c[{"Left": (-1, 0), "Right": (1, 2), "Up": (0, 0)}])
    print(ct[{"Left": (-1, 0), "Right": (1, 2), "Up": (0, 0)}])

    d = TAT(float, "Fermi")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}], True).range(1)
    print(d)
    dt = d.transpose(["Right", "Up", "Left"])
    print(dt)

    e = TAT(float)(["Down", "Up", "Left", "Right"], [2, 3, 4, 5]).range(1)
    print(e)
    et = e.transpose(["Left", "Down", "Right", "Up"])
    print(et)
    print(e[{"Down": 1, "Up": 1, "Left": 2, "Right": 2}])
    print(et[{"Down": 1, "Up": 1, "Left": 2, "Right": 2}])

    f = TAT(float)(["l1", "l2", "l3"], [2, 3, 4]).range()
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

    a = TAT(float)(["Left", "Right"], [2, 3]).set(initializer())
    b = a.merge_edge({"Merged": ["Left", "Right"]})
    c = a.merge_edge({"Merged": ["Right", "Left"]})
    d = c.split_edge({"Merged": [("1", 3), ("2", 2)]})
    print(a)
    print(b)
    print(c)
    print(d)

    e = TAT(complex, "Fermi")(["Left", "Right", "Up"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 2, 1: 3}, {-1: 2, 0: 3, 1: 1}]).set(initializer(0))
    print(e)
    f = e.merge_edge({"Merged": ["Left", "Up"]})
    print(f)
    g = f.split_edge({"Merged": [("Left", {-1: 3, 0: 1, 1: 2}), ("Up", {-1: 2, 0: 3, 1: 1})]})
    print(g)
    h = g.transpose(["Left", "Right", "Up"])
    print(h)


def test_edge_operator():
    print(TAT(float)(["A", "B"], [8, 8]).range())
    print(TAT(float)(["A", "B"], [8, 8]).range().edge_operator({"A": "C"}, {"C": [("D", 4), ("E", 2)], "B": [("F", 2), ("G", 4)]}, {"D", "F"}, {"I": ["D", "F"], "J": ["G", "E"]}, ["J", "I"]))
    print(TAT(float)(["A", "B", "C"], [2, 3, 4]).range().edge_operator({}, {}, set(), {}, ["B", "C", "A"]))

    a = TAT(float, "U1")(["Left", "Right", "Up", "Down"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 4, 1: 2}, {-1: 2, 0: 3, 1: 1}, {-1: 1, 0: 3, 1: 2}]).range(1)
    b = a.edge_rename({"Right": "Right1"}).split_edge({"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]})
    c = b.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_operator({"Right": "Right1"}, {"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]}, set(), {"Left": ["Left", "Down2"]}, ["Down1", "Right1", "Up", "Left"])
    print((total - d).norm_max())

    a = TAT(float, "Fermi")(["Left", "Right", "Up", "Down"], [{-1: 3, 0: 1, 1: 2}, {-1: 1, 0: 4, 1: 2}, {-1: 2, 0: 3, 1: 1}, {-1: 1, 0: 3, 1: 2}]).range(1)
    b = a.edge_rename({"Right": "Right1"}).split_edge({"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]})
    r = b.reverse_edge({"Left"})
    c = r.transpose(["Down1", "Right1", "Up", "Left", "Down2"])
    d = c.merge_edge({"Left": ["Left", "Down2"]})
    total = a.edge_operator({"Right": "Right1"}, {"Down": [("Down1", {0: 1, 1: 2}), ("Down2", {-1: 1, 0: 1})]}, {"Left"}, {"Left": ["Left", "Down2"]}, ["Down1", "Right1", "Up", "Left"])
    print((total - d).norm_max())
    print(total)


def test_contract():
    a = TAT(float)(["A", "B"], [2, 2]).range()
    b = TAT(float)(["C", "D"], [2, 2]).range()
    print(a)
    print(b)
    print(a.contract(b, {("A", "C")}))
    print(a.contract(b, {("A", "D")}))
    print(a.contract(b, {("B", "C")}))
    print(a.contract(b, {("B", "D")}))

    print(TAT(float)(["A", "B", "C", "D"], [1, 2, 3, 4]).range().contract(TAT(float)(["E", "F", "G", "H"], [3, 1, 2, 4]).range(), {("B", "G"), ("D", "H")}))

    c = TAT(float, "Fermi")(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).range()
    d = TAT(float, "Fermi")(["E", "F", "G", "H"], [{0: 2, 1: 1}, {-2: 1, -1: 1, 0: 2}, {0: 1, -1: 2}, {0: 2, 1: 1, 2: 2}], True).range()
    print(c)
    print(d)
    print(c.contract(d, {("B", "G"), ("D", "H")}))
    print(c.transpose(["A", "C", "B", "D"]).contract(d.transpose(["G", "H", "E", "F"]), {("B", "G"), ("D", "H")}))


def test_svd():
    a = TAT(float)(["A", "B", "C", "D"], [2, 3, 4, 5]).range()
    print(a)
    [u, s, v] = a.svd({"C", "A"}, "E", "F")
    print(u)
    print(v)
    print(s)
    print(v.multiple(s, "F", 'u').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]))

    b = TAT(complex)(["A", "B", "C", "D"], [2, 3, 4, 5]).range()
    print(b)
    [u, s, v] = b.svd({"A", "D"}, "E", "F")
    print(u)
    print(v)
    print(s)
    print(v.contract(v.edge_rename({"F": "F2"}), {("B", "B"), ("C", "C")}).transform(lambda i: i if abs(i) > 1e-5 else 0))
    print(v.multiple(s, "F", 'v').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]))

    c = TAT(float, "U1")(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).range()
    [u, s, v] = c.svd({"C", "A"}, "E", "F")
    print(u)
    print(s)
    print(v)
    print(c)
    print(v.multiple(s, "F", 'v').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))
    print(v.contract(u.multiple(s, "E", 'u'), {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))

    a = TAT(float)(["A", "B", "C", "D"], [2, 3, 4, 5]).range()
    print(a)
    [u, s, v] = a.svd({"C", "A"}, "E", "F", 2)
    print(u)
    print(v)
    print(s)
    print(v.multiple(s, "F", 'u').contract(u, {("F", "E")}).transpose(["A", "B", "C", "D"]).transform(lambda x: 0 if abs(x) < 0.01 else x))

    c = TAT(float, "U1")(["A", "B", "C", "D"], [{-1: 1, 0: 1, -2: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}, {-2: 2, -1: 1, 0: 2}], True).range()
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
