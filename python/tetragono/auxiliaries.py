#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from __future__ import annotations
from multimethod import multimethod
import lazy


def safe_contract(tensor_1: self.Tensor, tensor_2: self.Tensor, pair: set[tuple[str, str]]) -> self.Tensor:
    new_pair: set[tuple[str, str]] = set()
    for name_1, name_2 in pair:
        if name_1 in tensor_1.names and name_2 in tensor_2.names:
            new_pair.add((name_1, name_2))
    return tensor_1.contract(tensor_2, new_pair)


class Auxiliaries:

    def __init__(self, L1: int, L2: int, Dc: int, Tensor: type) -> None:
        self.L1: int = L1
        self.L2: int = L2
        self.cut_dimension: int = Dc
        self.Tensor: type = Tensor

        one: self.Tensor = self.Tensor(1)
        self._one: lazy.Node[self.Tensor] = lazy.Root(one)
        self._one_l1: lazy.Node[list[self.Tensor]] = lazy.Root([one for l1 in range(self.L1)])
        self._one_l2: lazy.Node[list[self.Tensor]] = lazy.Root([one for l2 in range(self.L2)])

        self._lattice: list[list[lazy.Node[self.Tensor]]] = [[lazy.Root() for l2 in range(self.L2)] for l1 in range(self.L1)]

        self._zip_row: list[lazy.Node[list[self.Tensor]]] = [lazy.Node(self._zip, *(self._lattice[l1][l2] for l2 in range(self.L2))) for l1 in range(self.L1)]
        self._up_to_down: dict[int, lazy.Node[list[self.Tensor]]] = {}
        self._up_to_down_site: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l1 in range(-1, self.L1):
            self._up_to_down[l1] = self._construct_up_to_down(l1)
            for l2 in range(self.L2):
                self._up_to_down_site[l1, l2] = lazy.Node(list.__getitem__, self._up_to_down[l1], l2)
        self._down_to_up: dict[int, lazy.Node[list[self.Tensor]]] = {}
        self._down_to_up_site: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l1 in reversed(range(self.L1 + 1)):
            self._down_to_up[l1] = self._construct_down_to_up(l1)
            for l2 in range(self.L2):
                self._down_to_up_site[l1, l2] = lazy.Node(list.__getitem__, self._down_to_up[l1], l2)

        self._zip_column: list[lazy.Node[list[self.Tensor]]] = [lazy.Node(self._zip, *(self._lattice[l1][l2] for l1 in range(self.L1))) for l2 in range(self.L2)]
        self._left_to_right: dict[int, lazy.Node[list[self.Tensor]]] = {}
        self._left_to_right_site: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l2 in range(-1, self.L2):
            self._left_to_right[l2] = self._construct_left_to_right(l2)
            for l1 in range(self.L1):
                self._left_to_right_site[l1, l2] = lazy.Node(list.__getitem__, self._left_to_right[l2], l1)
        self._right_to_left: dict[int, lazy.Node[list[self.Tensor]]] = {}
        self._right_to_left_site: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l2 in reversed(range(self.L2 + 1)):
            self._right_to_left[l2] = self._construct_right_to_left(l2)
            for l1 in range(self.L1):
                self._right_to_left_site[l1, l2] = lazy.Node(list.__getitem__, self._right_to_left[l2], l1)

        #   R1 -
        # > R2 -
        #   R3 -
        #   ^
        self._inline_left_to_right: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        #       DR1 -
        # > R2 - |
        #   R3 -
        #   ^
        self._inline_left_to_right_tailed: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l1 in range(self.L1):
            for l2 in range(-1, self.L2):
                self._inline_left_to_right[l1, l2] = self._construct_inline_left_to_right(l1, l2)
                self._inline_left_to_right_tailed[l1, l2] = self._construct_inline_left_to_right_tailed(l1, l2)

        # - l1
        # - L2 <
        # - L3
        #   ^
        self._inline_right_to_left: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        #      - L1
        #    | - L2 <
        # - UL3
        #        ^
        self._inline_right_to_left_tailed: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l1 in range(self.L1):
            for l2 in reversed(range(self.L2 + 1)):
                self._inline_right_to_left[l1, l2] = self._construct_inline_right_to_left(l1, l2)
                self._inline_right_to_left_tailed[l1, l2] = self._construct_inline_right_to_left_tailed(l1, l2)

        # D1 D2 D3 <
        # |  |  |
        #    ^
        self._inline_up_to_down: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        #     D2 D3 <
        #     |  |
        # RD1 -
        # |
        #     ^
        self._inline_up_to_down_tailed: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l2 in range(self.L2):
            for l1 in range(-1, self.L1):
                self._inline_up_to_down[l1, l2] = self._construct_inline_up_to_down(l1, l2)
                self._inline_up_to_down_tailed[l1, l2] = self._construct_inline_up_to_down_tailed(l1, l2)

        # |  |  |
        # U1 U2 U2 <
        #    ^
        self._inline_down_to_up: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        #       |
        #     - LU3
        # |  |
        # U1 U2     <
        #    ^
        self._inline_down_to_up_tailed: dict[tuple[int, int], lazy.Node[self.Tensor]] = {}
        for l2 in range(self.L2):
            for l1 in reversed(range(self.L1 + 1)):
                self._inline_down_to_up[l1, l2] = self._construct_inline_down_to_up(l1, l2)
                self._inline_down_to_up_tailed[l1, l2] = self._construct_inline_down_to_up_tailed(l1, l2)

    def __setitem__(self, l1l2: tuple[int, int], tensor: self.Tensor) -> None:
        l1, l2 = l1l2
        self._lattice[l1][l2].reset(tensor)

    def _construct_inline_left_to_right(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l2 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_left_to_right_in_lazy, self._inline_left_to_right_tailed[l1, l2 - 1], self._lattice[l1][l2], self._down_to_up_site[l1 + 1, l2], l1, l2)

    @staticmethod
    def _construct_inline_left_to_right_in_lazy(inline_left_to_right_tailed: self.Tensor, lattice: self.Tensor, down_to_up: self.Tensor, l1, l2) -> self.Tensor:
        # print("inline left to right", l1, l2)
        result: self.Tensor = safe_contract(inline_left_to_right_tailed, lattice.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")})
        result = safe_contract(result, down_to_up.edge_rename({"R": "R3"}), {("R3", "L"), ("D", "U")})
        return result

    def _construct_inline_left_to_right_tailed(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l2 == self.L2 - 1:
            return self._inline_left_to_right[l1, l2]
        else:
            return lazy.Node(self._construct_inline_left_to_right_tailed_in_lazy, self._inline_left_to_right[l1, l2], self._up_to_down_site[l1 - 1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_inline_left_to_right_tailed_in_lazy(inline_left_to_right: self.Tensor, up_to_down: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline left to right tailed", l1, l2)
        return safe_contract(inline_left_to_right, up_to_down, {("R1", "L")}).edge_rename({"R": "R1"})

    def _construct_inline_right_to_left(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l2 == self.L2:
            return self._one
        else:
            return lazy.Node(self._construct_inline_right_to_left_in_lazy, self._inline_right_to_left_tailed[l1, l2 + 1], self._lattice[l1][l2], self._up_to_down_site[l1 - 1, l2], l1, l2)

    @staticmethod
    def _construct_inline_right_to_left_in_lazy(inline_right_to_left_tailed: self.Tensor, lattice: self.Tensor, up_to_down: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline right to left", l1, l2)
        result: self.Tensor = safe_contract(inline_right_to_left_tailed, lattice.edge_rename({"L": "L2"}), {("L2", "R"), ("U", "D")})
        result = safe_contract(result, up_to_down.edge_rename({"L": "L1"}), {("L1", "R"), ("U", "D")})
        return result

    def _construct_inline_right_to_left_tailed(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l2 == 0:
            return self._inline_right_to_left[l1, l2]
        else:
            return lazy.Node(self._construct_inline_right_to_left_tailed_in_lazy, self._inline_right_to_left[l1, l2], self._down_to_up_site[l1 + 1, l2 - 1], l1, l2)

    @staticmethod
    def _construct_inline_right_to_left_tailed_in_lazy(inline_right_to_left: self.Tensor, down_to_up: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline right to left tailed", l1, l2)
        return safe_contract(inline_right_to_left, down_to_up, {("L3", "R")}).edge_rename({"L": "L3"})

    def _construct_inline_up_to_down(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l1 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_up_to_down_in_lazy, self._inline_up_to_down_tailed[l1 - 1, l2], self._lattice[l1][l2], self._right_to_left_site[l1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_inline_up_to_down_in_lazy(inline_up_to_down_tailed: self.Tensor, lattice: self.Tensor, right_to_left: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline up to down", l1, l2)
        result: self.Tensor = safe_contract(inline_up_to_down_tailed, lattice.edge_rename({"D": "D2"}), {("D2", "U"), ("R", "L")})
        result = safe_contract(result, right_to_left.edge_rename({"D": "D3"}), {("D3", "U"), ("R", "L")})
        return result

    def _construct_inline_up_to_down_tailed(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l1 == self.L1 - 1:
            return self._inline_left_to_right[l1, l2]
        else:
            return lazy.Node(self._construct_inline_up_to_down_tailed_in_lazy, self._inline_up_to_down[l1, l2], self._left_to_right_site[l1 + 1, l2 - 1], l1, l2)

    @staticmethod
    def _construct_inline_up_to_down_tailed_in_lazy(inline_up_to_down: self.Tensor, left_to_right: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline up to down tailed", l1, l2)
        return safe_contract(inline_up_to_down, left_to_right, {("D1", "U")}).edge_rename({"D": "D1"})

    def _construct_inline_down_to_up(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l1 == self.L1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_down_to_up_in_lazy, self._inline_down_to_up_tailed[l1 + 1, l2], self._lattice[l1][l2], self._left_to_right_site[l1, l2 - 1], l1, l2)

    @staticmethod
    def _construct_inline_down_to_up_in_lazy(inline_down_to_up_tailed: self.Tensor, lattice: self.Tensor, left_to_right: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline down to up", l1, l2)
        result: self.Tensor = safe_contract(inline_down_to_up_tailed, lattice.edge_rename({"U": "U2"}), {("U2", "D"), ("L", "R")})
        result = safe_contract(result, left_to_right.edge_rename({"U": "U1"}), {("U1", "D"), ("L", "R")})
        return result

    def _construct_inline_down_to_up_tailed(self, l1: int, l2: int) -> lazy.Node[self.Tensor]:
        if l1 == 0:
            return self._inline_down_to_up[l1, l2]
        else:
            return lazy.Node(self._construct_inline_down_to_up_tailed_in_lazy, self._inline_down_to_up[l1, l2], self._right_to_left_site[l1 - 1, l2 + 1], l1, l2)

    @staticmethod
    def _construct_inline_down_to_up_tailed_in_lazy(inline_down_to_up: self.Tensor, right_to_left: self.Tensor, l1: int, l2: int) -> self.Tensor:
        # print("inline down to up tailed", l1, l2)
        return safe_contract(inline_down_to_up, right_to_left, {("U3", "D")}).edge_rename({"U": "U3"})

    def _construct_up_to_down(self, l1: int) -> lazy.Node[list[self.Tensor]]:
        if l1 == -1:
            return self._one_l2
        else:
            return lazy.Node(self._two_line_to_one_line, "UDLR", self._up_to_down[l1 - 1], self._zip_row[l1], self.cut_dimension)

    def _construct_down_to_up(self, l1: int) -> lazy.Node[list[self.Tensor]]:
        if l1 == self.L1:
            return self._one_l2
        else:
            return lazy.Node(self._two_line_to_one_line, "DULR", self._down_to_up[l1 + 1], self._zip_row[l1], self.cut_dimension)

    def _construct_left_to_right(self, l2: int) -> lazy.Node[list[self.Tensor]]:
        if l2 == -1:
            return self._one_l1
        else:
            return lazy.Node(self._two_line_to_one_line, "LRUD", self._left_to_right[l2 - 1], self._zip_column[l2], self.cut_dimension)

    def _construct_right_to_left(self, l2: int) -> lazy.Node[list[self.Tensor]]:
        if l2 == self.L2:
            return self._one_l1
        else:
            return lazy.Node(self._two_line_to_one_line, "RLUD", self._right_to_left[l2 + 1], self._zip_column[l2], self.cut_dimension)

    @staticmethod
    def _zip(*args):
        return args

    @staticmethod
    def _two_line_to_one_line(udlr_name: list[str], line_1: list[self.Tensor], line_2: list[self.Tensor], cut: int) -> list[self.Tensor]:
        [up, down, left, right] = udlr_name
        up1 = up + "1"
        up2 = up + "2"
        down1 = down + "1"
        down2 = down + "2"
        left1 = left + "1"
        left2 = left + "2"
        right1 = right + "1"
        right2 = right + "2"

        length = len(line_1)
        if len(line_1) != len(line_2):
            raise ValueError("Different Length in Two Line to One Line")
        double_line = []
        for i in range(length):
            if down in line_1[i].names and up in line_2[i].names:
                pair = {(down, up)}
            else:
                pair = set()
            double_line.append(safe_contract(line_1[i].edge_rename({left: left1, right: right1}), line_2[i].edge_rename({left: left2, right: right2}), pair))

        for i in range(length - 1):
            q, r = double_line[i].qr("R", {right1, right2}, right, left)
            double_line[i] = q
            double_line[i + 1] = safe_contract(double_line[i + 1], r, {(left1, right1), (left2, right2)})

        for i in reversed(range(1, length)):
            [u, s, v] = double_line[i].svd({left}, right, left, left, right, cut)
            double_line[i] = v
            double_line[i - 1] = safe_contract(safe_contract(double_line[i - 1], u, {(right, left)}), s, {(right, left)})

        return double_line

    @multimethod
    def __call__(self, *, hint=None):  # -> self.Tensor
        if hint is None:
            hint = ("H", 0)
        direction, line = hint
        if direction == "H":
            return self._inline_right_to_left[line, 0]()
        elif directionn == "V":
            return self._inline_down_to_up[0, line]()
        else:
            raise ValueError("Unrecognized hint")

    @multimethod
    def __call__(self, position: tuple[tuple[int, int], ...], *, hint=None):  # -> self.Tensor
        if len(position) == 0:
            return self(hint=hint)
        elif len(position) == 1:
            l1, l2 = position[0]
            if hint is None:
                hint = "H"
            if hint == "H":
                left: self.Tensor = self._inline_left_to_right_tailed[l1, l2 - 1]()
                right: self.Tensor = self._inline_right_to_left_tailed[l1, l2 + 1]()
                big: self.Tensor = safe_contract(left, right, {("R1", "L1"), ("R3", "L3")})
                return big.edge_rename({"R2": "R0", "L2": "L0", "U": "U0", "D": "D0"})
            elif hint == "V":
                up: self.Tensor = self._inline_up_to_down_tailed[l1 - 1, l2]()
                down: self.Tensor = self._inline_up_to_down_tailed[l1 + 1, l2]()
                big: self.Tensor = safe_contract(up, down, {("U1", "D1"), ("U3", "D3")})
                return big.edge_rename({"U2": "U0", "D2": "D0", "R": "R0", "L": "L0"})
            else:
                raise ValueError("Unrecognized hint")
        elif len(position) == 2:
            p0, p1 = position
            if hint is not None:
                raise ValueError("Unrecognized hint")
            if p0[0] == p1[0]:
                if p0[1] + 1 == p1[1]:
                    left_part: self.Tensor = self._inline_left_to_right_tailed[p0[0], p0[1] - 1]()
                    left_dot: self.Tensor = self._down_to_up_site[p0[0] + 1, p0[1]]()
                    right_part: self.Tensor = self._inline_right_to_left_tailed[p1[0], p1[1] + 1]()
                    right_dot: self.Tensor = self._up_to_down_site[p1[0] - 1, p1[1]]()
                    result: self.Tensor = safe_contract(left_part, left_dot.edge_rename({"R": "R3"}), {("R3", "L")}).edge_rename({"D": "D0", "U": "U0"})
                    result = safe_contract(result, right_dot.edge_rename({"R": "R1"}), {("R1", "L")}).edge_rename({"D": "D1"})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R3", "L3")}).edge_rename({"U": "U1", "R2": "R0", "L2": "L1"})
                    return result
                if p0[1] == p1[1] + 1:
                    left_part: self.Tensor = self._inline_left_to_right_tailed[p1[0], p1[1] - 1]()
                    left_dot: self.Tensor = self._down_to_up_site[p1[0] + 1, p1[1]]()
                    right_part: self.Tensor = self._inline_right_to_left_tailed[p0[0], p0[1] + 1]()
                    right_dot: self.Tensor = self._up_to_down_site[p0[0] - 1, p0[1]]()
                    result: self.Tensor = safe_contract(left_part, left_dot.edge_rename({"R": "R3"}), {("R3", "L")}).edge_rename({"D": "D1", "U": "U1"})
                    result = safe_contract(result, right_dot.edge_rename({"R": "R1"}), {("R1", "L")}).edge_rename({"D": "D0"})
                    result = safe_contract(result, right_part, {("R1", "L1"), ("R3", "L3")}).edge_rename({"U": "U0", "R2": "R1", "L2": "L0"})
                    return result
            if p0[1] == p1[1]:
                if p0[0] + 1 == p1[0]:
                    up_part: self.Tensor = self._inline_up_to_down_tailed[p0[0] - 1, p0[1]]()
                    up_dot: self.Tensor = self._right_to_left_site[p0[0], p0[1] + 1]()
                    down_part: self.Tensor = self._inline_down_to_up_tailed[p1[0] + 1, p1[1]]()
                    down_dot: self.Tensor = self._left_to_right_site[p1[0], p1[1] - 1]()
                    result: self.Tensor = safe_contract(up_part, up_dot.edge_rename({"D": "D3"}), {("D3", "U")}).edge_rename({"R": "R0", "L": "L0"})
                    result = safe_contract(result, down_dot.edge_rename({"D": "D1"}), {("D1", "U")}).edge_rename({"R": "R1"})
                    result = safe_contract(result, down_part, {("D1", "U1"), ("D3", "U3")}).edge_rename({"L": "L1", "D2": "D0", "U2": "U1"})
                    return result
                if p0[0] == p1[0] + 1:
                    up_part: self.Tensor = self._inline_up_to_down_tailed[p1[0] - 1, p1[1]]()
                    up_dot: self.Tensor = self._right_to_left_site[p1[0], p1[1] + 1]()
                    down_part: self.Tensor = self._inline_down_to_up_tailed[p0[0] + 1, p0[1]]()
                    down_dot: self.Tensor = self._left_to_right_site[p0[0], p0[1] - 1]()
                    result: self.Tensor = safe_contract(up_part, up_dot.edge_rename({"D": "D3"}), {("D3", "U")}).edge_rename({"R": "R1", "L": "L1"})
                    result = safe_contract(result, down_dot.edge_rename({"D": "D1"}), {("D1", "U")}).edge_rename({"R": "R0"})
                    result = safe_contract(result, down_part, {("D1", "U1"), ("D3", "U3")}).edge_rename({"L": "L0", "D2": "D1", "U2": "U0"})
                    return result

        raise NotImplementedError("Unsupported auxilary hole style")

    # TODO double layer aux

    # TODO TRG aux

    # TODO contract with other tensor style __call__

    # TODO copy, __setstate__ __getstate__
