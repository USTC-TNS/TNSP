#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import lazy
from ..common_toolkit import safe_contract, safe_rename


class ThreeLineAuxiliaries:
    __slots__ = [
        "L2", "Tensor", "_ones", "_lattice_0n", "_lattice_0c", "_lattice_1n", "_lattice_1c", "_lattice_2",
        "_zip_column", "_left_to_right", "_right_to_left", "cut_dimension"
    ]

    def __init__(self, L2, Tensor, cut_dimension):
        self.L2 = L2
        self.Tensor = Tensor
        self.cut_dimension = cut_dimension

        one = self.Tensor(1)
        self._ones = lazy.Root([one for l1 in range(5)])

        self._lattice_0n = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_0c = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_1n = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_1c = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_2 = [lazy.Root() for l2 in range(self.L2)]

        self._zip_column = [
            lazy.Node(
                self._zip,
                self._lattice_0n[l2],
                self._lattice_1n[l2],
                self._lattice_2[l2],
                self._lattice_1c[l2],
                self._lattice_0c[l2],
            ) for l2 in range(self.L2)
        ]

        self._left_to_right = {}
        for l2 in range(-1, self.L2):
            self._left_to_right[l2] = self._construct_left_to_right(l2)

        self._right_to_left = {}
        for l2 in reversed(range(self.L2 + 1)):
            self._right_to_left[l2] = self._construct_right_to_left(l2)

    @staticmethod
    def _zip(*args):
        return args

    @staticmethod
    def _two_line_to_one_line(lr_name, line_1, line_2, cut):
        left, right = lr_name
        right_n = right + "N"
        right_c = right + "C"

        rn0, nr = line_2[0].qr('q', {q_name for q_name in (right,) if q_name in line_2[0].names}, "D", "U")
        n12 = safe_contract(safe_rename(nr, {"D": "D2"}), safe_rename(line_1[0], {"D": "D1"}), {(left, right)})
        n12 = safe_contract(n12, safe_rename(line_1[1], {"D": "D1"}), {("D1", "U")})
        n12 = safe_contract(n12, safe_rename(line_2[1], {"D": "D2"}), {(right, left), ("D2", "U")})

        rc0, cr = line_2[4].qr('q', {q_name for q_name in (right,) if q_name in line_2[0].names}, "D", "U")
        c12 = safe_contract(safe_rename(cr, {"D": "D2"}), safe_rename(line_1[4], {"D": "D1"}), {(left, right)})
        c12 = safe_contract(c12, safe_rename(line_1[3], {"D": "D1"}), {("D1", "U")})
        c12 = safe_contract(c12, safe_rename(line_2[3], {"D": "D2"}), {(right, left), ("D2", "U")})

        big = safe_contract(safe_rename(n12, {
            right: right_n,
            "U": "UN"
        }), safe_rename(line_1[2], {"UC": "UC1"}), {("D1", "UN")})
        big = safe_contract(big, safe_rename(line_2[2], {"UC": "UC2"}), {("D2", "UN"), (right, left)})
        big = safe_contract(big,
                            safe_rename(c12, {
                                right: right_c,
                                "U": "UC"
                            }), {("UC1", "D1"), ("UC2", "D2"), ("T", "T")},
                            contract_all_physics_edges=True)

        u, s, v = big.svd({"UN", right_n}, "D", "UN", "UN", "D", cut)
        rn1 = safe_rename(u, {"UN": "U", right_n: right})
        big = safe_contract(v, s, {("UN", "D")})
        u, s, v = big.svd({"UC", right_c}, "D", "UC", "UC", "D", cut)
        rc1 = safe_rename(u, {"UC": "U", right_c: right})
        big = safe_contract(v, s, {("UC", "D")})

        return [rn0, rn1, big, rc1, rc0]

    def _construct_left_to_right(self, l2):
        if l2 == -1:
            return self._ones
        else:
            return lazy.Node(self._two_line_to_one_line, "LR", self._left_to_right[l2 - 1], self._zip_column[l2],
                             self.cut_dimension)

    def _construct_right_to_left(self, l2):
        if l2 == self.L2:
            return self._ones
        else:
            return lazy.Node(self._two_line_to_one_line, "RL", self._right_to_left[l2 + 1], self._zip_column[l2],
                             self.cut_dimension)

    def hole(self, l2):
        n0 = self._lattice_0n[l2]()
        c0 = self._lattice_0c[l2]()
        n1 = self._lattice_1n[l2]()
        c1 = self._lattice_1c[l2]()
        t2 = self._lattice_2[l2]()

        n1 = safe_rename(n1, {name: "O" + name[1:] for name in n1.names if name.startswith("P")})
        c1 = safe_rename(c1, {name: "I" + name[1:] for name in c1.names if name.startswith("P")})

        line = [n0, n1, t2, c1, c0]

        left = self._left_to_right[l2 - 1]()
        right = self._right_to_left[l2 + 1]()

        result = safe_contract(safe_rename(left[0], {"D": "D1"}), safe_rename(line[0], {"D": "D2"}), {("R", "L")})
        result = safe_contract(result, safe_rename(right[0], {"D": "D3"}), {("R", "L")})

        result = safe_contract(result, safe_rename(left[1], {"D": "D1"}), {("D1", "U")})
        result = safe_contract(result, safe_rename(line[1], {"D": "D2"}), {("D2", "U"), ("R", "L")})
        result = safe_contract(result, safe_rename(right[1], {"D": "D3"}), {("D3", "U"), ("R", "L")})

        result = safe_contract(result, safe_rename(left[2], {"UC": "U1"}), {("D1", "UN")})
        result = safe_contract(result, safe_rename(line[2], {"UC": "U2"}), {("D2", "UN"), ("R", "L")})
        result = safe_contract(result, safe_rename(right[2], {"UC": "U3"}), {("D3", "UN"), ("R", "L")})

        result = safe_contract(result, safe_rename(left[3], {"U": "U1"}), {("U1", "D")})
        result = safe_contract(result, safe_rename(line[3], {"U": "U2"}), {("U2", "D"), ("R", "L"), ("T", "T")})
        result = safe_contract(result, safe_rename(right[3], {"U": "U3"}), {("U3", "D"), ("R", "L")})

        result = safe_contract(result, safe_rename(left[4], {"U": "U1"}), {("U1", "D")})
        result = safe_contract(result,
                               safe_rename(line[4], {"U": "U2"}), {("U2", "D"), ("R", "L"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, safe_rename(right[4], {"U": "U3"}), {("U3", "D"), ("R", "L")})

        return result

    def __setitem__(self, positions, tensor):
        if positions[0] == 2:
            l1, l2 = positions
            if l1 != 2:
                raise ValueError("Invalid position when setting lattice")
            self._lattice_2[l2].reset(tensor)
        else:
            l1, l2, nc = positions
            if nc in ("N", "n"):
                if l1 == 0:
                    self._lattice_0n[l2].reset(tensor)
                elif l1 == 1:
                    self._lattice_1n[l2].reset(tensor)
                else:
                    raise ValueError("Invalid position when setting lattice")
            elif nc in ("C", "c"):
                if l1 == 0:
                    self._lattice_0c[l2].reset(tensor)
                elif l1 == 1:
                    self._lattice_1c[l2].reset(tensor)
                else:
                    raise ValueError("Invalid position when setting lattice")
            else:
                raise ValueError("Invalid layer when setting lattice")
