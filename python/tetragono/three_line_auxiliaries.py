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

from __future__ import annotations
import lazy
from .auxiliaries import safe_contract, safe_rename


class ThreeLineAuxiliaries:
    __slots__ = [
        "L2", "Tensor", "_one", "_lattice_0n", "_lattice_0c", "_lattice_1n", "_lattice_1c", "_lattice_2",
        "_left_to_right", "_right_to_left"
    ]

    def __init__(self, L2, Tensor):
        self.L2 = L2
        self.Tensor = Tensor

        one = self.Tensor(1)
        self._one = lazy.Root(one)

        self._lattice_0n = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_0c = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_1n = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_1c = [lazy.Root() for l2 in range(self.L2)]
        self._lattice_2 = [lazy.Root() for l2 in range(self.L2)]

        self._left_to_right = {}
        for l2 in range(-1, self.L2):
            self._left_to_right[l2] = self._construct_left_to_right(l2)

        self._right_to_left = {}
        for l2 in reversed(range(self.L2 + 1)):
            self._right_to_left[l2] = self._construct_right_to_left(l2)

    def _construct_left_to_right(self, l2):
        if l2 == -1:
            return self._one
        else:
            return lazy.Node(
                self._construct_left_to_right_in_lazy,
                self._left_to_right[l2 - 1],
                self._lattice_0n[l2],
                self._lattice_0c[l2],
                self._lattice_1n[l2],
                self._lattice_1c[l2],
                self._lattice_2[l2],
                l2,
            )

    def _construct_right_to_left(self, l2):
        if l2 == self.L2:
            return self._one
        else:
            return lazy.Node(
                self._construct_right_to_left_in_lazy,
                self._right_to_left[l2 + 1],
                self._lattice_0n[l2],
                self._lattice_0c[l2],
                self._lattice_1n[l2],
                self._lattice_1c[l2],
                self._lattice_2[l2],
                l2,
            )

    @staticmethod
    def _construct_left_to_right_in_lazy(left, n0, c0, n1, c1, t2, l2):
        result = safe_contract(left, safe_rename(c0, {"R": "RC0"}), {("RC0", "L")})
        result = safe_contract(result, safe_rename(c1, {"R": "RC1"}), {("RC1", "L"), ("D", "U")})
        result = safe_contract(result, safe_rename(t2, {"R": "R2"}), {("R2", "L"), ("D", "UC")})
        result = safe_contract(result,
                               safe_rename(n1, {"R": "RN1"}), {("RN1", "L"), ("UN", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result,
                               safe_rename(n0, {"R": "RN0"}), {("RN0", "L"), ("U", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result /= result.norm_sum()
        return result

    @staticmethod
    def _construct_right_to_left_in_lazy(right, n0, c0, n1, c1, t2, l2):
        result = safe_contract(right, safe_rename(n0, {"L": "LN0"}), {("LN0", "R")})
        result = safe_contract(result, safe_rename(n1, {"L": "LN1"}), {("LN1", "R"), ("D", "U")})
        result = safe_contract(result, safe_rename(t2, {"L": "L2"}), {("L2", "R"), ("D", "UN")})
        result = safe_contract(result,
                               safe_rename(c1, {"L": "LC1"}), {("LC1", "R"), ("UC", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result,
                               safe_rename(c0, {"L": "LC0"}), {("LC0", "R"), ("U", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result /= result.norm_sum()
        return result

    def hole(self, l2, orbit):
        n0 = self._lattice_0n[l2]()
        c0 = self._lattice_0c[l2]()
        n1 = self._lattice_1n[l2]()
        c1 = self._lattice_1c[l2]()
        t2 = self._lattice_2[l2]()

        n1 = safe_rename(n1, {f"P{orbit}": f"O0"})
        c1 = safe_rename(c1, {f"P{orbit}": f"I0"})

        left = self._left_to_right[l2 - 1]()
        right = self._right_to_left[l2 + 1]()

        result = safe_contract(left, safe_rename(c0, {"R": "RC0"}), {("RC0", "L")})
        result = safe_contract(result, safe_rename(c1, {"R": "RC1"}), {("RC1", "L"), ("D", "U")})
        result = safe_contract(result, safe_rename(t2, {"R": "R2"}), {("R2", "L"), ("D", "UC")})
        result = safe_contract(result,
                               safe_rename(n1, {"R": "RN1"}), {("RN1", "L"), ("UN", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result,
                               safe_rename(n0, {"R": "RN0"}), {("RN0", "L"), ("U", "D"), ("T", "T")},
                               contract_all_physics_edges=True)
        result = safe_contract(result, right, {("RN0", "LN0"), ("RC0", "LC0"), ("RN1", "LN1"), ("RC1", "LC1"),
                                               ("R2", "L2")})
        result /= result.norm_sum()
        return result

    def __setitem__(self, positions, tensor):
        if positions[0] == 2:
            l1, l2 = positions
            if l1 != 2:
                raise ValueError("Invalid position when setting lattice")
            self._lattice_2[l2].reset(tensor)
        else:
            l1, l2, nc = positions
            if nc == "N" or nc == "n":
                if l1 == 0:
                    self._lattice_0n[l2].reset(tensor)
                elif l1 == 1:
                    self._lattice_1n[l2].reset(tensor)
                else:
                    raise ValueError("Invalid position when setting lattice")
            elif nc == "C" or nc == "c":
                if l1 == 0:
                    self._lattice_0c[l2].reset(tensor)
                elif l1 == 1:
                    self._lattice_1c[l2].reset(tensor)
                else:
                    raise ValueError("Invalid position when setting lattice")
            else:
                raise ValueError("Invalid layer when setting lattice")
