#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


class SingleLayerAuxiliaries:
    __slots__ = [
        "L1", "L2", "cut_dimension", "normalize", "Tensor", "_one", "_one_l1", "_one_l2", "_lattice", "_zip_row",
        "_up_to_down", "_up_to_down_site", "_down_to_up", "_down_to_up_site", "_zip_column", "_left_to_right",
        "_left_to_right_site", "_right_to_left", "_right_to_left_site", "_inline_left_to_right",
        "_inline_left_to_right_tailed", "_inline_right_to_left", "_inline_right_to_left_tailed", "_inline_up_to_down",
        "_inline_up_to_down_tailed", "_inline_down_to_up", "_inline_down_to_up_tailed", "_4_inline_left_to_right",
        "_4_inline_left_to_right_tailed", "_4_inline_right_to_left", "_4_inline_right_to_left_tailed"
    ]

    def copy(self, cp=None):
        result = self.__new__(type(self))
        if cp is None:
            cp = lazy.Copy()

        result.L1 = self.L1
        result.L2 = self.L2
        result.cut_dimension = self.cut_dimension
        result.normalize = self.normalize
        result.Tensor = self.Tensor

        result._one = cp(self._one)
        result._one_l1 = cp(self._one_l1)
        result._one_l2 = cp(self._one_l2)

        result._lattice = [[cp(root) for root in row] for row in self._lattice]

        result._zip_row = [cp(row) for row in self._zip_row]
        result._up_to_down = {k: cp(v) for k, v in self._up_to_down.items()}
        result._up_to_down_site = {k: cp(v) for k, v in self._up_to_down_site.items()}
        result._down_to_up = {k: cp(v) for k, v in self._down_to_up.items()}
        result._down_to_up_site = {k: cp(v) for k, v in self._down_to_up_site.items()}

        result._zip_column = [cp(column) for column in self._zip_column]
        result._left_to_right = {k: cp(v) for k, v in self._left_to_right.items()}
        result._left_to_right_site = {k: cp(v) for k, v in self._left_to_right_site.items()}
        result._right_to_left = {k: cp(v) for k, v in self._right_to_left.items()}
        result._right_to_left_site = {k: cp(v) for k, v in self._right_to_left_site.items()}

        result._inline_left_to_right = {}
        result._inline_left_to_right_tailed = {}
        for l1 in range(self.L1):
            for l2 in range(-1, self.L2):
                result._inline_left_to_right[l1, l2] = cp(self._inline_left_to_right[l1, l2])
                result._inline_left_to_right_tailed[l1, l2] = cp(self._inline_left_to_right_tailed[l1, l2])

        result._inline_right_to_left = {}
        result._inline_right_to_left_tailed = {}
        for l1 in range(self.L1):
            for l2 in reversed(range(self.L2 + 1)):
                result._inline_right_to_left[l1, l2] = cp(self._inline_right_to_left[l1, l2])
                result._inline_right_to_left_tailed[l1, l2] = cp(self._inline_right_to_left_tailed[l1, l2])

        result._inline_up_to_down = {}
        result._inline_up_to_down_tailed = {}
        for l2 in range(self.L2):
            for l1 in range(-1, self.L1):
                result._inline_up_to_down[l1, l2] = cp(self._inline_up_to_down[l1, l2])
                result._inline_up_to_down_tailed[l1, l2] = cp(self._inline_up_to_down_tailed[l1, l2])

        result._inline_down_to_up = {}
        result._inline_down_to_up_tailed = {}
        for l2 in range(self.L2):
            for l1 in reversed(range(self.L1 + 1)):
                result._inline_down_to_up[l1, l2] = cp(self._inline_down_to_up[l1, l2])
                result._inline_down_to_up_tailed[l1, l2] = cp(self._inline_down_to_up_tailed[l1, l2])

        result._4_inline_left_to_right = {}
        result._4_inline_left_to_right_tailed = {}
        for l1 in range(self.L1 - 1):
            for l2 in range(-1, self.L2):
                result._4_inline_left_to_right[l1, l2] = cp(self._4_inline_left_to_right[l1, l2])
                result._4_inline_left_to_right_tailed[l1, l2] = cp(self._4_inline_left_to_right_tailed[l1, l2])

        result._4_inline_right_to_left = {}
        result._4_inline_right_to_left_tailed = {}
        for l1 in range(self.L1 - 1):
            for l2 in reversed(range(self.L2 + 1)):
                result._4_inline_right_to_left[l1, l2] = cp(self._4_inline_right_to_left[l1, l2])
                result._4_inline_right_to_left_tailed[l1, l2] = cp(self._4_inline_right_to_left_tailed[l1, l2])

        return result

    def __init__(self, L1, L2, cut_dimension, normalize, Tensor):
        self.L1 = L1
        self.L2 = L2
        self.cut_dimension = cut_dimension
        self.normalize = normalize
        self.Tensor = Tensor

        one = self.Tensor(1)
        self._one = lazy.Root(one)
        self._one_l1 = lazy.Root([one for l1 in range(self.L1)])
        self._one_l2 = lazy.Root([one for l2 in range(self.L2)])

        self._lattice = [[lazy.Root() for l2 in range(self.L2)] for l1 in range(self.L1)]

        self._zip_row = [
            lazy.Node(self._zip, *(self._lattice[l1][l2] for l2 in range(self.L2))) for l1 in range(self.L1)
        ]
        self._up_to_down = {}
        self._up_to_down_site = {}
        for l1 in range(-1, self.L1):
            self._up_to_down[l1] = self._construct_up_to_down(l1)
            for l2 in range(self.L2):
                self._up_to_down_site[l1, l2] = lazy.Node(list.__getitem__, self._up_to_down[l1], l2)
        self._down_to_up = {}
        self._down_to_up_site = {}
        for l1 in reversed(range(self.L1 + 1)):
            self._down_to_up[l1] = self._construct_down_to_up(l1)
            for l2 in range(self.L2):
                self._down_to_up_site[l1, l2] = lazy.Node(list.__getitem__, self._down_to_up[l1], l2)

        self._zip_column = [
            lazy.Node(self._zip, *(self._lattice[l1][l2] for l1 in range(self.L1))) for l2 in range(self.L2)
        ]
        self._left_to_right = {}
        self._left_to_right_site = {}
        for l2 in range(-1, self.L2):
            self._left_to_right[l2] = self._construct_left_to_right(l2)
            for l1 in range(self.L1):
                self._left_to_right_site[l1, l2] = lazy.Node(list.__getitem__, self._left_to_right[l2], l1)
        self._right_to_left = {}
        self._right_to_left_site = {}
        for l2 in reversed(range(self.L2 + 1)):
            self._right_to_left[l2] = self._construct_right_to_left(l2)
            for l1 in range(self.L1):
                self._right_to_left_site[l1, l2] = lazy.Node(list.__getitem__, self._right_to_left[l2], l1)

        #   R1 -
        # > R2 -
        #   R3 -
        #   ^
        self._inline_left_to_right = {}
        #       DR1 -
        # > R2 - |
        #   R3 -
        #   ^
        self._inline_left_to_right_tailed = {}
        for l1 in range(self.L1):
            for l2 in range(-1, self.L2):
                self._inline_left_to_right[l1, l2] = self._construct_inline_left_to_right(l1, l2)
                self._inline_left_to_right_tailed[l1, l2] = self._construct_inline_left_to_right_tailed(l1, l2)

        # - l1
        # - L2 <
        # - L3
        #   ^
        self._inline_right_to_left = {}
        #      - L1
        #    | - L2 <
        # - UL3
        #        ^
        self._inline_right_to_left_tailed = {}
        for l1 in range(self.L1):
            for l2 in reversed(range(self.L2 + 1)):
                self._inline_right_to_left[l1, l2] = self._construct_inline_right_to_left(l1, l2)
                self._inline_right_to_left_tailed[l1, l2] = self._construct_inline_right_to_left_tailed(l1, l2)

        # D1 D2 D3 <
        # |  |  |
        #    ^
        self._inline_up_to_down = {}
        #     D2 D3 <
        #     |  |
        # RD1 -
        # |
        #     ^
        self._inline_up_to_down_tailed = {}
        for l2 in range(self.L2):
            for l1 in range(-1, self.L1):
                self._inline_up_to_down[l1, l2] = self._construct_inline_up_to_down(l1, l2)
                self._inline_up_to_down_tailed[l1, l2] = self._construct_inline_up_to_down_tailed(l1, l2)

        # |  |  |
        # U1 U2 U2 <
        #    ^
        self._inline_down_to_up = {}
        #       |
        #     - LU3
        # |  |
        # U1 U2     <
        #    ^
        self._inline_down_to_up_tailed = {}
        for l2 in range(self.L2):
            for l1 in reversed(range(self.L1 + 1)):
                self._inline_down_to_up[l1, l2] = self._construct_inline_down_to_up(l1, l2)
                self._inline_down_to_up_tailed[l1, l2] = self._construct_inline_down_to_up_tailed(l1, l2)

        #   R1 -
        # > R2 -
        #   R3 -
        #   R4 -
        #   ^
        self._4_inline_left_to_right = {}
        #       DR1 -
        # > R2 - |
        #   R3 -
        #   R4 -
        #   ^
        self._4_inline_left_to_right_tailed = {}
        for l1 in range(self.L1 - 1):
            for l2 in range(-1, self.L2):
                self._4_inline_left_to_right[l1, l2] = self._construct_4_inline_left_to_right(l1, l2)
                self._4_inline_left_to_right_tailed[l1, l2] = self._construct_4_inline_left_to_right_tailed(l1, l2)

        # - l1
        # - L2 <
        # - L3
        # - L4
        #   ^
        self._4_inline_right_to_left = {}
        #      - L1
        #      - L2 <
        #    | - L3
        # - UL4
        #        ^
        self._4_inline_right_to_left_tailed = {}
        for l1 in range(self.L1 - 1):
            for l2 in reversed(range(self.L2 + 1)):
                self._4_inline_right_to_left[l1, l2] = self._construct_4_inline_right_to_left(l1, l2)
                self._4_inline_right_to_left_tailed[l1, l2] = self._construct_4_inline_right_to_left_tailed(l1, l2)

    def __setitem__(self, l1l2, tensor):
        l1, l2 = l1l2
        self._lattice[l1][l2].reset(tensor)

    def __getitem__(self, l1l2):
        l1, l2 = l1l2
        return self._lattice[l1][l2]()

    def _construct_inline_left_to_right(self, l1, l2):
        if l2 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_left_to_right_in_lazy, self._inline_left_to_right_tailed[l1,
                                                                                                             l2 - 1],
                             self._lattice[l1][l2], self._down_to_up_site[l1 + 1, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_left_to_right_in_lazy(inline_left_to_right_tailed, lattice, down_to_up, normalize, l1, l2):
        # print("inline left to right", l1, l2)
        result = safe_contract(inline_left_to_right_tailed, safe_rename(lattice, {"R": "R2"}), {("R2", "L"),
                                                                                                ("D", "U")})
        result = safe_contract(result, safe_rename(down_to_up, {"R": "R3"}), {("R3", "L"), ("D", "U")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_inline_left_to_right_tailed(self, l1, l2):
        if l2 == self.L2 - 1:
            return self._inline_left_to_right[l1, l2]
        else:
            return lazy.Node(self._construct_inline_left_to_right_tailed_in_lazy, self._inline_left_to_right[l1, l2],
                             self._up_to_down_site[l1 - 1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_inline_left_to_right_tailed_in_lazy(inline_left_to_right, up_to_down, l1, l2):
        # print("inline left to right tailed", l1, l2)
        return safe_rename(safe_contract(inline_left_to_right, up_to_down, {("R1", "L")}), {"R": "R1"})

    def _construct_inline_right_to_left(self, l1, l2):
        if l2 == self.L2:
            return self._one
        else:
            return lazy.Node(self._construct_inline_right_to_left_in_lazy, self._inline_right_to_left_tailed[l1,
                                                                                                             l2 + 1],
                             self._lattice[l1][l2], self._up_to_down_site[l1 - 1, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_right_to_left_in_lazy(inline_right_to_left_tailed, lattice, up_to_down, normalize, l1, l2):
        # print("inline right to left", l1, l2)
        result = safe_contract(inline_right_to_left_tailed, safe_rename(lattice, {"L": "L2"}), {("L2", "R"),
                                                                                                ("U", "D")})
        result = safe_contract(result, safe_rename(up_to_down, {"L": "L1"}), {("L1", "R"), ("U", "D")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_inline_right_to_left_tailed(self, l1, l2):
        if l2 == 0:
            return self._inline_right_to_left[l1, l2]
        else:
            return lazy.Node(self._construct_inline_right_to_left_tailed_in_lazy, self._inline_right_to_left[l1, l2],
                             self._down_to_up_site[l1 + 1, l2 - 1], l1, l2)

    @staticmethod
    def _construct_inline_right_to_left_tailed_in_lazy(inline_right_to_left, down_to_up, l1, l2):
        # print("inline right to left tailed", l1, l2)
        return safe_rename(safe_contract(inline_right_to_left, down_to_up, {("L3", "R")}), {"L": "L3"})

    def _construct_inline_up_to_down(self, l1, l2):
        if l1 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_up_to_down_in_lazy, self._inline_up_to_down_tailed[l1 - 1, l2],
                             self._lattice[l1][l2], self._right_to_left_site[l1, l2 + 1], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_up_to_down_in_lazy(inline_up_to_down_tailed, lattice, right_to_left, normalize, l1, l2):
        # print("inline up to down", l1, l2)
        result = safe_contract(inline_up_to_down_tailed, safe_rename(lattice, {"D": "D2"}), {("D2", "U"), ("R", "L")})
        result = safe_contract(result, safe_rename(right_to_left, {"D": "D3"}), {("D3", "U"), ("R", "L")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_inline_up_to_down_tailed(self, l1, l2):
        if l1 == self.L1 - 1:
            return self._inline_left_to_right[l1, l2]
        else:
            return lazy.Node(self._construct_inline_up_to_down_tailed_in_lazy, self._inline_up_to_down[l1, l2],
                             self._left_to_right_site[l1 + 1, l2 - 1], l1, l2)

    @staticmethod
    def _construct_inline_up_to_down_tailed_in_lazy(inline_up_to_down, left_to_right, l1, l2):
        # print("inline up to down tailed", l1, l2)
        return safe_rename(safe_contract(inline_up_to_down, left_to_right, {("D1", "U")}), {"D": "D1"})

    def _construct_inline_down_to_up(self, l1, l2):
        if l1 == self.L1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_down_to_up_in_lazy, self._inline_down_to_up_tailed[l1 + 1, l2],
                             self._lattice[l1][l2], self._left_to_right_site[l1, l2 - 1], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_down_to_up_in_lazy(inline_down_to_up_tailed, lattice, left_to_right, normalize, l1, l2):
        # print("inline down to up", l1, l2)
        result = safe_contract(inline_down_to_up_tailed, safe_rename(lattice, {"U": "U2"}), {("U2", "D"), ("L", "R")})
        result = safe_contract(result, safe_rename(left_to_right, {"U": "U1"}), {("U1", "D"), ("L", "R")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_inline_down_to_up_tailed(self, l1, l2):
        if l1 == 0:
            return self._inline_down_to_up[l1, l2]
        else:
            return lazy.Node(self._construct_inline_down_to_up_tailed_in_lazy, self._inline_down_to_up[l1, l2],
                             self._right_to_left_site[l1 - 1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_inline_down_to_up_tailed_in_lazy(inline_down_to_up, right_to_left, l1, l2):
        # print("inline down to up tailed", l1, l2)
        return safe_rename(safe_contract(inline_down_to_up, right_to_left, {("U3", "D")}), {"U": "U3"})

    def _construct_4_inline_left_to_right(self, l1, l2):
        if l2 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_4_inline_left_to_right_in_lazy,
                             self._4_inline_left_to_right_tailed[l1, l2 - 1], self._lattice[l1][l2],
                             self._lattice[l1 + 1][l2], self._down_to_up_site[l1 + 2, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_4_inline_left_to_right_in_lazy(inline_left_to_right_tailed, lattice_2, lattice_3, down_to_up,
                                                  normalize, l1, l2):
        result = safe_contract(inline_left_to_right_tailed, safe_rename(lattice_2, {"R": "R2"}), {("R2", "L"),
                                                                                                  ("D", "U")})
        result = safe_contract(result, safe_rename(lattice_3, {"R": "R3"}), {("R3", "L"), ("D", "U")})
        result = safe_contract(result, safe_rename(down_to_up, {"R": "R4"}), {("R4", "L"), ("D", "U")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_4_inline_left_to_right_tailed(self, l1, l2):
        if l2 == self.L2 - 1:
            return self._4_inline_left_to_right[l1, l2]
        else:
            return lazy.Node(self._construct_4_inline_left_to_right_tailed_in_lazy,
                             self._4_inline_left_to_right[l1, l2], self._up_to_down_site[l1 - 1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_4_inline_left_to_right_tailed_in_lazy(inline_left_to_right, up_to_down, l1, l2):
        return safe_rename(safe_contract(inline_left_to_right, up_to_down, {("R1", "L")}), {"R": "R1"})

    def _construct_4_inline_right_to_left(self, l1, l2):
        if l2 == self.L2:
            return self._one
        else:
            return lazy.Node(self._construct_4_inline_right_to_left_in_lazy,
                             self._4_inline_right_to_left_tailed[l1, l2 + 1], self._lattice[l1 + 1][l2],
                             self._lattice[l1][l2], self._up_to_down_site[l1 - 1, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_4_inline_right_to_left_in_lazy(inline_right_to_left_tailed, lattice_3, lattice_2, up_to_down,
                                                  normalize, l1, l2):
        result = safe_contract(inline_right_to_left_tailed, safe_rename(lattice_3, {"L": "L3"}), {("L3", "R"),
                                                                                                  ("U", "D")})
        result = safe_contract(result, safe_rename(lattice_2, {"L": "L2"}), {("L2", "R"), ("U", "D")})
        result = safe_contract(result, safe_rename(up_to_down, {"L": "L1"}), {("L1", "R"), ("U", "D")})
        if normalize:
            result /= result.norm_2()
        return result

    def _construct_4_inline_right_to_left_tailed(self, l1, l2):
        if l2 == 0:
            return self._4_inline_right_to_left[l1, l2]
        else:
            return lazy.Node(self._construct_4_inline_right_to_left_tailed_in_lazy,
                             self._4_inline_right_to_left[l1, l2], self._down_to_up_site[l1 + 2, l2 - 1], l1, l2)

    @staticmethod
    def _construct_4_inline_right_to_left_tailed_in_lazy(inline_right_to_left, down_to_up, l1, l2):
        return safe_rename(safe_contract(inline_right_to_left, down_to_up, {("L4", "R")}), {"L": "L4"})

    def _construct_up_to_down(self, l1):
        if l1 == -1:
            return self._one_l2
        else:
            return lazy.Node(self._two_line_to_one_line, "UDLR", self._up_to_down[l1 - 1], self._zip_row[l1],
                             self.cut_dimension, self.normalize)

    def _construct_down_to_up(self, l1):
        if l1 == self.L1:
            return self._one_l2
        else:
            return lazy.Node(self._two_line_to_one_line, "DULR", self._down_to_up[l1 + 1], self._zip_row[l1],
                             self.cut_dimension, self.normalize)

    def _construct_left_to_right(self, l2):
        if l2 == -1:
            return self._one_l1
        else:
            return lazy.Node(self._two_line_to_one_line, "LRUD", self._left_to_right[l2 - 1], self._zip_column[l2],
                             self.cut_dimension, self.normalize)

    def _construct_right_to_left(self, l2):
        if l2 == self.L2:
            return self._one_l1
        else:
            return lazy.Node(self._two_line_to_one_line, "RLUD", self._right_to_left[l2 + 1], self._zip_column[l2],
                             self.cut_dimension, self.normalize)

    @staticmethod
    def _zip(*args):
        return args

    @staticmethod
    def _two_line_to_one_line(udlr_name, line_1, line_2, cut, normalize):
        [up, down, left, right] = udlr_name
        left1 = left + "1"
        left2 = left + "2"
        right1 = right + "1"
        right2 = right + "2"

        length = len(line_1)
        if len(line_1) != len(line_2):
            raise ValueError("Different Length in Two Line to One Line")
        double_line = []
        for i in range(length):
            double_line.append(
                safe_contract(safe_rename(line_1[i], {
                    left: left1,
                    right: right1
                }), safe_rename(line_2[i], {
                    left: left2,
                    right: right2
                }), {(down, up)}))

        for i in range(length - 1):
            q, r = double_line[i].qr('r', {r_name for r_name in (right1, right2) if r_name in double_line[i].names},
                                     right, left)
            double_line[i] = q
            double_line[i + 1] = safe_contract(double_line[i + 1], r, {(left1, right1), (left2, right2)})

        for i in reversed(range(1, length)):
            [u, s, v] = double_line[i].svd({left}, right, left, left, right, cut)
            if normalize:
                s /= s.norm_2()
            double_line[i] = v
            double_line[i - 1] = safe_contract(safe_contract(double_line[i - 1], u, {(right, left)}), s,
                                               {(right, left)})

        return double_line

    def replace(self, replacement, *, hint=None):
        if len(replacement) == 0:
            if hint is None:
                hint = ("H", 0)
            direction, line = hint
            if direction == "H":
                return self._inline_right_to_left[line, 0]()
            elif direction == "V":
                return self._inline_down_to_up[0, line]()
            else:
                raise ValueError("Unrecognized hint")

        minl1 = min(l1 for l1, l2 in replacement.keys())
        minl2 = min(l2 for l1, l2 in replacement.keys())
        maxl1 = max(l1 for l1, l2 in replacement.keys())
        maxl2 = max(l2 for l1, l2 in replacement.keys())

        if maxl1 - minl1 == 0 and maxl2 - minl2 == 0:
            l1 = minl1
            l2 = minl2
            new_tensor = replacement[l1, l2]
            if self._inline_left_to_right_tailed[l1, l2 - 1] and self._inline_right_to_left_tailed[l1, l2 + 1]:
                left = self._inline_left_to_right_tailed[l1, l2 - 1]()
                right = self._inline_right_to_left_tailed[l1, l2 + 1]()
                result = safe_contract(left, new_tensor, {("R2", "L"), ("D", "U")})
                result = safe_contract(result, right, {("R1", "L1"), ("R", "L2"), ("D", "U"), ("R3", "L3")})
                return result
            if self._inline_up_to_down_tailed[l1 - 1, l2] and self._inline_down_to_up_tailed[l1 + 1, l2]:
                up = self._inline_up_to_down_tailed[l1 - 1, l2]()
                down = self._inline_down_to_up_tailed[l1 + 1, l2]()
                result = safe_contract(up, new_tensor, {("D2", "U"), ("R", "L")})
                result = safe_contract(result, down, {("D1", "U1"), ("D", "U2"), ("R", "L"), ("D3", "U3")})
                return result
            if l1 != self.L1 - 1:
                if self._4_inline_left_to_right_tailed[l1, l2 - 1] and self._4_inline_right_to_left_tailed[l1, l2 + 1]:
                    left_part = self._4_inline_left_to_right_tailed[l1, l2 - 1]()
                    right_part = self._4_inline_right_to_left_tailed[l1, l2 + 1]()
                    another_tensor = self._lattice[l1 + 1][l2]()
                    result = safe_contract(left_part, safe_rename(new_tensor, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                    result = safe_contract(result, safe_rename(another_tensor, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                                ("R4", "L4")})
                    return result
            if l1 != 0:
                if self._4_inline_left_to_right_tailed[l1 - 1, l2 - 1] and self._4_inline_right_to_left_tailed[l1 - 1,
                                                                                                               l2 + 1]:
                    left_part = self._4_inline_left_to_right_tailed[l1 - 1, l2 - 1]()
                    right_part = self._4_inline_right_to_left_tailed[l1 - 1, l2 + 1]()
                    another_tensor = self._lattice[l1 - 1][l2]()
                    result = safe_contract(left_part, safe_rename(another_tensor, {"R": "R2"}), {("D", "U"),
                                                                                                 ("R2", "L")})
                    result = safe_contract(result, safe_rename(new_tensor, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                                ("R4", "L4")})
                    return result
            if hint is None:
                hint = "H"
            if hint == "H":
                left = self._inline_left_to_right_tailed[l1, l2 - 1]()
                right = self._inline_right_to_left_tailed[l1, l2 + 1]()
                result = safe_contract(left, new_tensor, {("R2", "L"), ("D", "U")})
                result = safe_contract(result, right, {("R1", "L1"), ("R", "L2"), ("D", "U"), ("R3", "L3")})
                return result
            elif hint == "V":
                up = self._inline_up_to_down_tailed[l1 - 1, l2]()
                down = self._inline_down_to_up_tailed[l1 + 1, l2]()
                result = safe_contract(up, new_tensor, {("D2", "U"), ("R", "L")})
                result = safe_contract(result, down, {("D1", "U1"), ("D", "U2"), ("R", "L"), ("D3", "U3")})
                return result
            else:
                raise ValueError("Unrecognized hint")

        if hint is not None:
            raise ValueError("Unrecognized hint")
        if maxl1 - minl1 == 0 and maxl2 - minl2 == 1:
            # left right
            items = list(replacement.items())
            p0, t0 = items[0]
            p1, t1 = items[1]
            if p1[1] < p0[1]:
                p0, t0, p1, t1 = p1, t1, p0, t0
            l1, l2 = p0
            if (self._inline_left_to_right_tailed[l1, l2 - 1] and self._down_to_up_site[l1 + 1, l2] and
                    self._inline_right_to_left_tailed[l1, l2 + 2] and self._up_to_down_site[l1 - 1, l2 + 1]):
                left_part = self._inline_left_to_right_tailed[l1, l2 - 1]()
                left_dot = self._down_to_up_site[l1 + 1, l2]()
                right_part = self._inline_right_to_left_tailed[l1, l2 + 2]()
                right_dot = self._up_to_down_site[l1 - 1, l2 + 1]()
                result = safe_contract(left_part, safe_rename(t0, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                result = safe_contract(result, safe_rename(left_dot, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
                result = safe_contract(result, safe_rename(t1, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("D", "U"), ("R3", "L3")})
                return result
            if l1 != self.L1 - 1:
                if (self._4_inline_left_to_right_tailed[l1, l2 - 1] and self._down_to_up_site[l1 + 2, l2] and
                        self._4_inline_right_to_left_tailed[l1, l2 + 2] and self._up_to_down_site[l1 - 1, l2 + 1]):
                    left_part = self._4_inline_left_to_right_tailed[l1, l2 - 1]()
                    left_dot = self._down_to_up_site[l1 + 2, l2]()
                    right_part = self._4_inline_right_to_left_tailed[l1, l2 + 2]()
                    right_dot = self._up_to_down_site[l1 - 1, l2 + 1]()
                    t0_down = self._lattice[l1 + 1][l2]()
                    t1_down = self._lattice[l1 + 1][l2 + 1]()
                    result = safe_contract(left_part, safe_rename(t0, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                    result = safe_contract(result, safe_rename(t0_down, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, safe_rename(left_dot, {"R": "R4"}), {("D", "U"), ("R4", "L")})
                    result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
                    result = safe_contract(result, safe_rename(t1, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                    result = safe_contract(result, safe_rename(t1_down, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                                ("R4", "L4")})
                    return result
            if l1 != 0:
                if (self._4_inline_left_to_right_tailed[l1 - 1, l2 - 1] and self._down_to_up_site[l1 + 1, l2] and
                        self._4_inline_right_to_left_tailed[l1 - 1, l2 + 2] and self._up_to_down_site[l1 - 2, l2 + 1]):
                    left_part = self._4_inline_left_to_right_tailed[l1 - 1, l2 - 1]()
                    left_dot = self._down_to_up_site[l1 + 1, l2]()
                    right_part = self._4_inline_right_to_left_tailed[l1 - 1, l2 + 2]()
                    right_dot = self._up_to_down_site[l1 - 2, l2 + 1]()
                    t0_up = self._lattice[l1 - 1][l2]()
                    t1_up = self._lattice[l1 - 1][l2 + 1]()
                    result = safe_contract(left_part, safe_rename(t0_up, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                    result = safe_contract(result, safe_rename(t0, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, safe_rename(left_dot, {"R": "R4"}), {("D", "U"), ("R4", "L")})
                    result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
                    result = safe_contract(result, safe_rename(t1_up, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                    result = safe_contract(result, safe_rename(t1, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                                ("R4", "L4")})
                    return result
            left_part = self._inline_left_to_right_tailed[l1, l2 - 1]()
            left_dot = self._down_to_up_site[l1 + 1, l2]()
            right_part = self._inline_right_to_left_tailed[l1, l2 + 2]()
            right_dot = self._up_to_down_site[l1 - 1, l2 + 1]()
            result = safe_contract(left_part, safe_rename(t0, {"R": "R2"}), {("D", "U"), ("R2", "L")})
            result = safe_contract(result, safe_rename(left_dot, {"R": "R3"}), {("D", "U"), ("R3", "L")})
            result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
            result = safe_contract(result, safe_rename(t1, {"R": "R2"}), {("D", "U"), ("R2", "L")})
            result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("D", "U"), ("R3", "L3")})
            return result
        if maxl1 - minl1 == 1 and maxl2 - minl2 == 0:
            # up
            # down
            items = list(replacement.items())
            p0, t0 = items[0]
            p1, t1 = items[1]
            if p1[0] < p0[0]:
                p0, t0, p1, t1 = p1, t1, p0, t0
            l1, l2 = p0
            if (self._inline_up_to_down_tailed[l1 - 1, l2] and self._right_to_left_site[l1, l2 + 1] and
                    self._inline_down_to_up_tailed[l1 + 2, l2] and self._left_to_right_site[l1 + 1, l2 - 1]):
                up_part = self._inline_up_to_down_tailed[l1 - 1, l2]()
                up_dot = self._right_to_left_site[l1, l2 + 1]()
                down_part = self._inline_down_to_up_tailed[l1 + 2, l2]()
                down_dot = self._left_to_right_site[l1 + 1, l2 - 1]()
                result = safe_contract(up_part, safe_rename(t0, {"D": "D2"}), {("R", "L"), ("D2", "U")})
                result = safe_contract(result, safe_rename(up_dot, {"D": "D3"}), {("R", "L"), ("D3", "U")})
                result = safe_contract(result, safe_rename(down_dot, {"D": "D1"}), {("D1", "U")})
                result = safe_contract(result, safe_rename(t1, {"D": "D2"}), {("R", "L"), ("D2", "U")})
                result = safe_contract(result, down_part, {("D1", "U1"), ("D2", "U2"), ("R", "L"), ("D3", "U3")})
                return result
            if self._4_inline_left_to_right_tailed[l1, l2 - 1] and self._4_inline_right_to_left_tailed[l1, l2 + 1]:
                left_part = self._4_inline_left_to_right_tailed[l1, l2 - 1]()
                right_part = self._4_inline_right_to_left_tailed[l1, l2 + 1]()

                result = safe_contract(left_part, safe_rename(t0, {"R": "R2"}), {("D", "U"), ("R2", "L")})
                result = safe_contract(result, safe_rename(t1, {"R": "R3"}), {("D", "U"), ("R3", "L")})
                result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                            ("R4", "L4")})
                return result
            up_part = self._inline_up_to_down_tailed[l1 - 1, l2]()
            up_dot = self._right_to_left_site[l1, l2 + 1]()
            down_part = self._inline_down_to_up_tailed[l1 + 2, l2]()
            down_dot = self._left_to_right_site[l1 + 1, l2 - 1]()
            result = safe_contract(up_part, safe_rename(t0, {"D": "D2"}), {("R", "L"), ("D2", "U")})
            result = safe_contract(result, safe_rename(up_dot, {"D": "D3"}), {("R", "L"), ("D3", "U")})
            result = safe_contract(result, safe_rename(down_dot, {"D": "D1"}), {("D1", "U")})
            result = safe_contract(result, safe_rename(t1, {"D": "D2"}), {("R", "L"), ("D2", "U")})
            result = safe_contract(result, down_part, {("D1", "U1"), ("D2", "U2"), ("R", "L"), ("D3", "U3")})
            return result

        if maxl1 - minl1 == 1 and maxl2 - minl2 == 1:
            # t11 t12
            # t21 t22
            t = [[self._lattice[minl1][minl2](), self._lattice[minl1][maxl2]()],
                 [self._lattice[maxl1][minl2](), self._lattice[maxl1][maxl2]()]]
            for [l1, l2], tensor in replacement.items():
                t[l1 - minl1][l2 - minl2] = tensor

            l1 = minl1
            l2 = minl2

            left_part = self._4_inline_left_to_right_tailed[l1, l2 - 1]()
            left_dot = self._down_to_up_site[l1 + 2, l2]()
            right_part = self._4_inline_right_to_left_tailed[l1, l2 + 2]()
            right_dot = self._up_to_down_site[l1 - 1, l2 + 1]()

            result = safe_contract(left_part, safe_rename(t[0][0], {"R": "R2"}), {("D", "U"), ("R2", "L")})
            result = safe_contract(result, safe_rename(t[1][0], {"R": "R3"}), {("D", "U"), ("R3", "L")})
            result = safe_contract(result, safe_rename(left_dot, {"R": "R4"}), {("D", "U"), ("R4", "L")})
            result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
            result = safe_contract(result, safe_rename(t[0][1], {"R": "R2"}), {("D", "U"), ("R2", "L")})
            result = safe_contract(result, safe_rename(t[1][1], {"R": "R3"}), {("D", "U"), ("R3", "L")})
            result = safe_contract(result, right_part, {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U"),
                                                        ("R4", "L4")})
            return result

        # If not implemented, return None instead of raise error.
        return None

    def hole(self, position, *, hint=None):
        if len(position) == 0:
            if hint is None:
                hint = ("H", 0)
            direction, line = hint
            if direction == "H":
                return self._inline_right_to_left[line, 0]()
            elif direction == "V":
                return self._inline_down_to_up[0, line]()
            else:
                raise ValueError("Unrecognized hint")
        elif len(position) == 1:
            l1, l2 = position[0]
            if hint is None:
                hint = "H"
            if hint == "H":
                left = self._inline_left_to_right_tailed[l1, l2 - 1]()
                right = self._inline_right_to_left_tailed[l1, l2 + 1]()
                big = safe_contract(left, right, {("R1", "L1"), ("R3", "L3")})
                return safe_rename(big, {"R2": "R0", "L2": "L0", "U": "U0", "D": "D0"})
            elif hint == "V":
                up = self._inline_up_to_down_tailed[l1 - 1, l2]()
                down = self._inline_down_to_up_tailed[l1 + 1, l2]()
                big = safe_contract(up, down, {("U1", "D1"), ("U3", "D3")})
                return safe_rename(big, {"U2": "U0", "D2": "D0", "R": "R0", "L": "L0"})
            else:
                raise ValueError("Unrecognized hint")
        elif len(position) == 2:
            p0, p1 = position
            if hint is not None:
                raise ValueError("Unrecognized hint")
            if p0[0] == p1[0]:
                if p0[1] + 1 == p1[1]:
                    left_part = self._inline_left_to_right_tailed[p0[0], p0[1] - 1]()
                    left_dot = self._down_to_up_site[p0[0] + 1, p0[1]]()
                    right_part = self._inline_right_to_left_tailed[p1[0], p1[1] + 1]()
                    right_dot = self._up_to_down_site[p1[0] - 1, p1[1]]()
                    result = safe_rename(safe_contract(left_part, safe_rename(left_dot, {"R": "R3"}), {("R3", "L")}), {
                        "D": "D0",
                        "U": "U0"
                    })
                    result = safe_rename(safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")}),
                                         {"D": "D1"})
                    result = safe_rename(safe_contract(result, right_part, {("R1", "L1"), ("R3", "L3")}), {
                        "U": "U1",
                        "R2": "R0",
                        "L2": "L1"
                    })
                    return result
                if p0[1] == p1[1] + 1:
                    left_part = self._inline_left_to_right_tailed[p1[0], p1[1] - 1]()
                    left_dot = self._down_to_up_site[p1[0] + 1, p1[1]]()
                    right_part = self._inline_right_to_left_tailed[p0[0], p0[1] + 1]()
                    right_dot = self._up_to_down_site[p0[0] - 1, p0[1]]()
                    result = safe_rename(safe_contract(left_part, safe_rename(left_dot, {"R": "R3"}), {("R3", "L")}), {
                        "D": "D1",
                        "U": "U1"
                    })
                    result = safe_rename(safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")}),
                                         {"D": "D0"})
                    result = safe_rename(safe_contract(result, right_part, {("R1", "L1"), ("R3", "L3")}), {
                        "U": "U0",
                        "R2": "R1",
                        "L2": "L0"
                    })
                    return result
            if p0[1] == p1[1]:
                if p0[0] + 1 == p1[0]:
                    up_part = self._inline_up_to_down_tailed[p0[0] - 1, p0[1]]()
                    up_dot = self._right_to_left_site[p0[0], p0[1] + 1]()
                    down_part = self._inline_down_to_up_tailed[p1[0] + 1, p1[1]]()
                    down_dot = self._left_to_right_site[p1[0], p1[1] - 1]()
                    result = safe_rename(safe_contract(up_part, safe_rename(up_dot, {"D": "D3"}), {("D3", "U")}), {
                        "R": "R0",
                        "L": "L0"
                    })
                    result = safe_rename(safe_contract(result, safe_rename(down_dot, {"D": "D1"}), {("D1", "U")}),
                                         {"R": "R1"})
                    result = safe_rename(safe_contract(result, down_part, {("D1", "U1"), ("D3", "U3")}), {
                        "L": "L1",
                        "D2": "D0",
                        "U2": "U1"
                    })
                    return result
                if p0[0] == p1[0] + 1:
                    up_part = self._inline_up_to_down_tailed[p1[0] - 1, p1[1]]()
                    up_dot = self._right_to_left_site[p1[0], p1[1] + 1]()
                    down_part = self._inline_down_to_up_tailed[p0[0] + 1, p0[1]]()
                    down_dot = self._left_to_right_site[p0[0], p0[1] - 1]()
                    result = safe_rename(safe_contract(up_part, safe_rename(up_dot, {"D": "D3"}), {("D3", "U")}), {
                        "R": "R1",
                        "L": "L1"
                    })
                    result = safe_rename(safe_contract(result, safe_rename(down_dot, {"D": "D1"}), {("D1", "U")}),
                                         {"R": "R0"})
                    result = safe_rename(safe_contract(result, down_part, {("D1", "U1"), ("D3", "U3")}), {
                        "L": "L0",
                        "D2": "D1",
                        "U2": "U0"
                    })
                    return result

        raise NotImplementedError("Unsupported auxilary hole style")
