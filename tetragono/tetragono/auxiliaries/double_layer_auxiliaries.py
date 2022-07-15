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


class DoubleLayerAuxiliaries:
    __slots__ = [
        "L1", "L2", "cut_dimension", "normalize", "Tensor", "_one", "_one_l1", "_one_l2", "_lattice_n", "_lattice_c",
        "_zip_row", "_up_to_down", "_up_to_down_site", "_down_to_up", "_down_to_up_site", "_zip_column",
        "_left_to_right", "_left_to_right_site", "_right_to_left", "_right_to_left_site", "_inline_left_to_right",
        "_inline_left_to_right_tailed", "_inline_right_to_left", "_inline_right_to_left_tailed", "_inline_up_to_down",
        "_inline_up_to_down_tailed", "_inline_down_to_up", "_inline_down_to_up_tailed"
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

        result._lattice_n = [[cp(root) for root in row] for row in self._lattice_n]
        result._lattice_c = [[cp(root) for root in row] for row in self._lattice_c]

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

        self._lattice_n = [[lazy.Root() for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._lattice_c = [[lazy.Root() for l2 in range(self.L2)] for l1 in range(self.L1)]

        self._zip_row = [
            lazy.Node(self._zip, self.L2, *(self._lattice_n[l1][l2] for l2 in range(self.L2)),
                      *(self._lattice_c[l1][l2] for l2 in range(self.L2))) for l1 in range(self.L1)
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
            lazy.Node(self._zip, self.L1, *(self._lattice_n[l1][l2] for l1 in range(self.L1)),
                      *(self._lattice_c[l1][l2] for l1 in range(self.L1))) for l2 in range(self.L2)
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

    def __setitem__(self, l1l2nc, tensor):
        l1, l2, nc = l1l2nc
        if nc in ("N", "n"):
            self._lattice_n[l1][l2].reset(tensor)
        elif nc in ("C", "c"):
            self._lattice_c[l1][l2].reset(tensor)
        else:
            raise ValueError("Invalid layer when setting lattice")

    def __getitem__(self, l1l2nc):
        l1, l2, nc = l1l2nc
        if nc in ("N", "n"):
            return self._lattice_n[l1][l2]()
        elif nc in ("C", "c"):
            return self._lattice_c[l1][l2]()
        else:
            raise ValueError("Invalid layer when setting lattice")

    # X2 -> XN + XC, where X = L R U D

    def _construct_inline_left_to_right(self, l1, l2):
        if l2 == -1:
            return self._one
        else:
            return lazy.Node(self._construct_inline_left_to_right_in_lazy,
                             self._inline_left_to_right_tailed[l1, l2 - 1], self._lattice_n[l1][l2],
                             self._lattice_c[l1][l2], self._down_to_up_site[l1 + 1, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_left_to_right_in_lazy(inline_left_to_right_tailed, lattice_n, lattice_c, down_to_up,
                                                normalize, l1, l2):
        # print("inline left to right", l1, l2)
        result = safe_contract(inline_left_to_right_tailed, safe_rename(lattice_n, {
            "R": "RN",
            "D": "DN"
        }), {("RN", "L"), ("DN", "U")})
        result = safe_contract(result,
                               safe_rename(lattice_c, {
                                   "R": "RC",
                                   "D": "DC"
                               }), {("RC", "L"), ("DC", "U"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, safe_rename(down_to_up, {"R": "R3"}), {("R3", "L"), ("DN", "UN"), ("DC", "UC")})
        if normalize:
            result /= result.norm_sum()
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
            return lazy.Node(self._construct_inline_right_to_left_in_lazy,
                             self._inline_right_to_left_tailed[l1, l2 + 1], self._lattice_n[l1][l2],
                             self._lattice_c[l1][l2], self._up_to_down_site[l1 - 1, l2], self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_right_to_left_in_lazy(inline_right_to_left_tailed, lattice_n, lattice_c, up_to_down,
                                                normalize, l1, l2):
        # print("inline right to left", l1, l2)
        result = safe_contract(inline_right_to_left_tailed, safe_rename(lattice_n, {
            "L": "LN",
            "U": "UN"
        }), {("LN", "R"), ("UN", "D")})
        result = safe_contract(result,
                               safe_rename(lattice_c, {
                                   "L": "LC",
                                   "U": "UC"
                               }), {("LC", "R"), ("UC", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, safe_rename(up_to_down, {"L": "L1"}), {("L1", "R"), ("UN", "DN"), ("UC", "DC")})
        if normalize:
            result /= result.norm_sum()
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
                             self._lattice_n[l1][l2], self._lattice_c[l1][l2], self._right_to_left_site[l1, l2 + 1],
                             self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_up_to_down_in_lazy(inline_up_to_down_tailed, lattice_n, lattice_c, right_to_left, normalize,
                                             l1, l2):
        # print("inline up to down", l1, l2)
        result = safe_contract(inline_up_to_down_tailed, safe_rename(lattice_n, {
            "D": "DN",
            "R": "RN"
        }), {("DN", "U"), ("RN", "L")})
        result = safe_contract(result,
                               safe_rename(lattice_c, {
                                   "D": "DC",
                                   "R": "RC"
                               }), {("DC", "U"), ("RC", "L"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, safe_rename(right_to_left, {"D": "D3"}), {("D3", "U"), ("RN", "LN"),
                                                                                 ("RC", "LC")})
        if normalize:
            result /= result.norm_sum()
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
                             self._lattice_n[l1][l2], self._lattice_c[l1][l2], self._left_to_right_site[l1, l2 - 1],
                             self.normalize, l1, l2)

    @staticmethod
    def _construct_inline_down_to_up_in_lazy(inline_down_to_up_tailed, lattice_n, lattice_c, left_to_right, normalize,
                                             l1, l2):
        # print("inline down to up", l1, l2)
        result = safe_contract(inline_down_to_up_tailed, safe_rename(lattice_n, {
            "U": "UN",
            "L": "LN"
        }), {("UN", "D"), ("LN", "R")})
        result = safe_contract(result,
                               safe_rename(lattice_c, {
                                   "U": "UC",
                                   "L": "LC"
                               }), {("UC", "D"), ("LC", "R"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, safe_rename(left_to_right, {"U": "U1"}), {("U1", "D"), ("LN", "RN"),
                                                                                 ("LC", "RC")})
        if normalize:
            result /= result.norm_sum()
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
    def _zip(length, *args):
        result = []
        for i in range(length):
            result.append((args[i], args[i + length]))
        return result

    @staticmethod
    def _two_line_to_one_line(udlr_name, line_1, line_2, cut, normalize):
        [up, down, left, right] = udlr_name
        down_n = down + "N"
        down_c = down + "C"
        left0 = left + "0"
        left1 = left + "1"
        right0 = right + "0"
        right1 = right + "1"

        length = len(line_1)
        if len(line_1) != len(line_2):
            raise ValueError("Different Length in Two Line to One Line")

        # do approximation in two stage which can reduce the time complexity.
        # contract line_1 with one part of line_2 first then, contract to another part is faster
        # than contracting two parts of line_2 first, then contract into line_1.
        # line_1[i]: L, R, DN, DC
        # line_2[i][0]: L, R, U, D, P
        # line_2[i][1]: L, R, U, D, P

        # Stage 1:
        stage_1 = []
        for i in range(length):
            this_site = safe_contract(
                safe_rename(line_1[i], {
                    left: left1,
                    right: right1
                }),
                safe_rename(line_2[i][0], {
                    left: left0,
                    right: right0,
                    down: down_n
                }),
                {(down_n, up)},
            )

            stage_1.append(this_site)

        for i in range(length - 1):
            q, r = stage_1[i].qr('r', {r_name for r_name in (right1, right0) if r_name in stage_1[i].names}, right,
                                 left)
            stage_1[i] = q
            stage_1[i + 1] = safe_contract(stage_1[i + 1], r, {(left1, right1), (left0, right0)})

        for i in reversed(range(1, length)):
            [u, s, v] = stage_1[i].svd({left}, right, left, left, right, cut)
            if normalize:
                s /= s.norm_sum()
            stage_1[i] = v
            stage_1[i - 1] = safe_contract(safe_contract(stage_1[i - 1], u, {(right, left)}), s, {(right, left)})

        # Stage 2:
        stage_2 = []
        for i in range(length):
            this_site = safe_contract(
                safe_rename(stage_1[i], {
                    left: left1,
                    right: right1
                }),
                safe_rename(line_2[i][1], {
                    left: left0,
                    right: right0,
                    down: down_c
                }),
                {("T", "T"), (down_c, up)},
                contract_all_physics_edges=True,
            )
            stage_2.append(this_site)
        for i in range(length - 1):
            q, r = stage_2[i].qr('r', {r_name for r_name in (right1, right0) if r_name in stage_2[i].names}, right,
                                 left)
            stage_2[i] = q
            stage_2[i + 1] = safe_contract(stage_2[i + 1], r, {(left1, right1), (left0, right0)})

        for i in reversed(range(1, length)):
            [u, s, v] = stage_2[i].svd({left}, right, left, left, right, cut)
            if normalize:
                s /= s.norm_sum()
            stage_2[i] = v
            stage_2[i - 1] = safe_contract(safe_contract(stage_2[i - 1], u, {(right, left)}), s, {(right, left)})
        return stage_2

    def hole(self, positions, *, hint=None):
        coordinates = []
        index_and_orbit = []
        for l1, l2, orbit in positions:
            if (l1, l2) not in coordinates:
                coordinates.append((l1, l2))
            index = coordinates.index((l1, l2))
            index_and_orbit.append((index, orbit))
        result = self._hole(coordinates, index_and_orbit, hint=hint)
        return result

    def _hole(self, coordinates, index_and_orbit, *, hint=None):
        if len(coordinates) == 0:
            if hint is None:
                hint = ("H", 0)
            direction, line = hint
            if direction == "H":
                return self._inline_right_to_left[line, 0]()
            elif direction == "V":
                return self._inline_down_to_up[0, line]()
            else:
                raise ValueError("Unrecognized hint")
        elif len(coordinates) == 1:
            l1, l2 = coordinates[0]
            if hint is None:
                hint = "H"
            if hint == "H":
                left = self._inline_left_to_right_tailed[l1, l2 - 1]()
                right = self._inline_right_to_left_tailed[l1, l2 + 1]()
                result = safe_contract(
                    left,
                    safe_rename(
                        self._lattice_n[l1][l2](), {
                            "R": "RN",
                            "D": "DN",
                            **{
                                f"P{orbit}": f"O{body_index}" for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                            }
                        }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_c[l1][l2](), {
                            "R": "RC",
                            "D": "DC",
                            **{
                                f"P{orbit}": f"I{body_index}" for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                            }
                        }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, right, {("R1", "L1"), ("R3", "L3"), ("RN", "LN"), ("RC", "LC"),
                                                       ("DN", "UN"), ("DC", "UC")})
                return result
            elif hint == "V":
                up = self._inline_up_to_down_tailed[l1 - 1, l2]()
                down = self._inline_down_to_up_tailed[l1 + 1, l2]()
                result = safe_contract(
                    up,
                    safe_rename(
                        self._lattice_n[l1][l2](), {
                            "R": "RN",
                            "D": "DN",
                            **{
                                f"P{orbit}": f"O{body_index}" for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                            }
                        }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_c[l1][l2](), {
                            "R": "RC",
                            "D": "DC",
                            **{
                                f"P{orbit}": f"I{body_index}" for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                            }
                        }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, down, {("D1", "U1"), ("D3", "U3"), ("RN", "LN"), ("RC", "LC"),
                                                      ("DN", "UN"), ("DC", "UC")})
                return result
            else:
                raise ValueError("Unrecognized hint")
        elif len(coordinates) == 2:
            p0, p1 = coordinates
            if hint is not None:
                raise ValueError("Unrecognized hint")
            if p0[0] == p1[0] and abs(p0[1] - p1[1]) == 1:
                if p0[1] + 1 == p1[1]:
                    i, j = p0
                    left_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                    ]
                    right_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
                    ]
                elif p0[1] == p1[1] + 1:
                    i, j = p1
                    left_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
                    ]
                    right_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                    ]
                else:
                    raise RuntimeError("Wrong double layer auxiliaries hole dispatch")
                left_part = self._inline_left_to_right_tailed[i, j - 1]()
                left_dot = self._down_to_up_site[i + 1, j]()
                right_part = self._inline_right_to_left_tailed[i, j + 2]()
                right_dot = self._up_to_down_site[i - 1, j + 1]()
                result = safe_contract(
                    left_part,
                    safe_rename(
                        self._lattice_n[i][j](), {
                            "R": "RN",
                            "D": "DN",
                            **{f"P{orbit}": f"O{body_index}" for body_index, orbit in left_index_and_orbit}
                        }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_c[i][j](), {
                            "R": "RC",
                            "D": "DC",
                            **{f"P{orbit}": f"I{body_index}" for body_index, orbit in left_index_and_orbit}
                        }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, safe_rename(left_dot, {"R": "R3"}), {("R3", "L"), ("DN", "UN"),
                                                                                    ("DC", "UC")})
                result = safe_contract(result, safe_rename(right_dot, {"R": "R1"}), {("R1", "L")})
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_n[i][j + 1](), {
                            "R": "RN",
                            "D": "DN",
                            **{f"P{orbit}": f"O{body_index}" for body_index, orbit in right_index_and_orbit}
                        }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_c[i][j + 1](), {
                            "R": "RC",
                            "D": "DC",
                            **{f"P{orbit}": f"I{body_index}" for body_index, orbit in right_index_and_orbit}
                        }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, right_part, {("R1", "L1"), ("R3", "L3"), ("RN", "LN"), ("RC", "LC"),
                                                            ("DN", "UN"), ("DC", "UC")})
                return result
            if p0[1] == p1[1] and abs(p0[0] - p1[0]) == 1:
                if p0[0] + 1 == p1[0]:
                    i, j = p0
                    up_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                    ]
                    down_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
                    ]
                elif p0[0] == p1[0] + 1:
                    i, j = p1
                    up_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
                    ]
                    down_index_and_orbit = [
                        (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
                    ]
                else:
                    raise RuntimeError("Wrong double layer auxiliaries hole dispatch")
                up_part = self._inline_up_to_down_tailed[i - 1, j]()
                up_dot = self._right_to_left_site[i, j + 1]()
                down_part = self._inline_down_to_up_tailed[i + 2, j]()
                down_dot = self._left_to_right_site[i + 1, j - 1]()
                result = safe_contract(
                    up_part,
                    safe_rename(self._lattice_n[i][j](), {
                        "R": "RN",
                        "D": "DN",
                        **{f"P{orbit}": f"O{body_index}" for body_index, orbit in up_index_and_orbit}
                    }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(self._lattice_c[i][j](), {
                        "R": "RC",
                        "D": "DC",
                        **{f"P{orbit}": f"I{body_index}" for body_index, orbit in up_index_and_orbit}
                    }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, safe_rename(up_dot, {"D": "D3"}), {("D3", "U"), ("RN", "LN"),
                                                                                  ("RC", "LC")})
                result = safe_contract(result, safe_rename(down_dot, {"D": "D1"}), {("D1", "U")})
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_n[i + 1][j](), {
                            "R": "RN",
                            "D": "DN",
                            **{f"P{orbit}": f"O{body_index}" for body_index, orbit in down_index_and_orbit}
                        }),
                    {("RN", "L"), ("DN", "U")},
                )
                result = safe_contract(
                    result,
                    safe_rename(
                        self._lattice_c[i + 1][j](), {
                            "R": "RC",
                            "D": "DC",
                            **{f"P{orbit}": f"I{body_index}" for body_index, orbit in down_index_and_orbit}
                        }),
                    {("RC", "L"), ("DC", "U"), ("T", "T")},
                    contract_all_physics_edges=True,
                )
                result = safe_contract(result, down_part, {("D1", "U1"), ("D3", "U3"), ("RN", "LN"), ("RC", "LC"),
                                                           ("DN", "UN"), ("DC", "UC")})
                return result
        raise NotImplementedError("Unsupported auxilary hole style")
