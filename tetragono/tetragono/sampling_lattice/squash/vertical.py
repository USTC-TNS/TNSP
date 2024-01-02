#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import numpy as np
from ...utility import safe_rename
from ...abstract_state import AbstractState
from ...abstract_lattice import AbstractLattice
from ..lattice import SamplingLattice


def unsquash(new_state, old_state, cut_dimension):
    # Create site map
    site_map = {}
    for l1, l2 in new_state.sites():
        orbit_index = 0
        for orbit in old_state.physics_edges[l1 * 2, l2]:
            site_map[l1 * 2, l2, orbit] = (l1, l2, orbit_index)
            orbit_index += 1
        for orbit in old_state.physics_edges[l1 * 2 + 1, l2]:
            site_map[l1 * 2 + 1, l2, orbit] = (l1, l2, orbit_index)
            orbit_index += 1
    # Set lattice tensor
    for l1, l2 in new_state.sites():
        new_tensor = new_state[l1, l2]
        part1 = old_state[l1 * 2, l2]
        part2 = old_state[l1 * 2 + 1, l2]
        left_split = []
        if "L" in part1.names:
            left_split.append(("L1", part1.edge_by_name("L").segments))
        if "L" in part2.names:
            left_split.append(("L2", part2.edge_by_name("L").segments))
        down_split = []
        if "R" in part1.names:
            down_split.append(("R1", part1.edge_by_name("R").segments))
        if "R" in part2.names:
            down_split.append(("R2", part2.edge_by_name("R").segments))
        split_plan = {}
        if left_split:
            split_plan["L"] = left_split
        if down_split:
            split_plan["R"] = down_split
        both = new_tensor.split_edge(split_plan, parity_exclude_name_split_set={"L"})
        new_part1, singular, new_part2 = both.svd(
            {"L1", "R1", "T", "U"} | {edge for edge in part1.names if edge.startswith("P")}, "D", "U", "U", "D",
            cut_dimension)
        identity = singular.same_shape().identity_({("U", "D")})
        delta = singular.sqrt()
        identity *= delta
        singular *= delta.reciprocal()
        new_part1 = safe_rename(new_part1, {
            "L1": "L",
            "R1": "R"
        } | {f"P{site_map[l1*2,l2,orbit][2]}": f"P{orbit}" for orbit in old_state.physics_edges[l1 * 2, l2]})
        new_part2 = safe_rename(new_part2, {
            "L2": "L",
            "R2": "R"
        } | {f"P{site_map[l1*2+1,l2,orbit][2]}": f"P{orbit}" for orbit in old_state.physics_edges[l1 * 2 + 1, l2]})
        old_state[l1 * 2, l2] = new_part1.contract(singular, {("D", "U")})
        old_state[l1 * 2 + 1, l2] = new_part2.contract(identity, {("U", "D")})
    # Update virtual bond
    for l1, l2 in old_state.sites():
        if l1 != old_state.L1 - 1:
            old_state.virtual_bond[l1, l2, "D"] = old_state[l1, l2].edge_by_name("D")
        if l2 != old_state.L2 - 1:
            old_state.virtual_bond[l1, l2, "R"] = old_state[l1, l2].edge_by_name("R")
    return old_state


def squash(old_state):
    new_state = AbstractState(old_state.Tensor, old_state.L1 // 2, old_state.L2)
    # Create site map
    site_map = {}
    for l1, l2 in new_state.sites():
        orbit_index = 0
        for orbit in old_state.physics_edges[l1 * 2, l2]:
            site_map[l1 * 2, l2, orbit] = (l1, l2, orbit_index)
            orbit_index += 1
        for orbit in old_state.physics_edges[l1 * 2 + 1, l2]:
            site_map[l1 * 2 + 1, l2, orbit] = (l1, l2, orbit_index)
            orbit_index += 1
    # Map physics edge
    for old, new in site_map.items():
        new_state.physics_edges[new] = old_state.physics_edges[old]
    # Map hamiltonian
    for positions, hamiltonian in old_state.hamiltonians:
        new_state.hamiltonians[[site_map[position] for position in positions]] = hamiltonian
    new_state.total_symmetry = old_state.total_symmetry
    new_state = AbstractLattice(new_state)
    # Prepare contracted tensor
    temporary_lattice = [[None for l2 in range(new_state.L2)] for l1 in range(new_state.L1)]
    for l1, l2 in new_state.sites():
        part1 = safe_rename(old_state[l1 * 2, l2], {
            "L": "L1",
            "R": "R1"
        } | {f"P{orbit}": f"P{site_map[l1*2,l2,orbit][2]}" for orbit in old_state.physics_edges[l1 * 2, l2]})
        part2 = safe_rename(old_state[l1 * 2 + 1, l2], {
            "L": "L2",
            "R": "R2"
        } | {f"P{orbit}": f"P{site_map[l1*2+1,l2,orbit][2]}" for orbit in old_state.physics_edges[l1 * 2 + 1, l2]})
        both = part1.contract(part2, {("D", "U")})
        left_merge = []
        if "L1" in both.names:
            left_merge.append("L1")
        if "L2" in both.names:
            left_merge.append("L2")
        right_merge = []
        if "R1" in both.names:
            right_merge.append("R1")
        if "R2" in both.names:
            right_merge.append("R2")
        merge_plan = {}
        if left_merge:
            merge_plan["L"] = left_merge
        if right_merge:
            merge_plan["R"] = right_merge
        temporary_lattice[l1][l2] = both.merge_edge(merge_plan, parity_exclude_name_merge_set={"L"})
    # Set virtual bond
    for l1, l2 in new_state.sites():
        if l2 != new_state.L2 - 1:
            new_state.virtual_bond[l1, l2, "R"] = temporary_lattice[l1][l2].edge_by_name("R")
        if l1 != new_state.L1 - 1:
            new_state.virtual_bond[l1, l2, "D"] = temporary_lattice[l1][l2].edge_by_name("D")
    new_state = SamplingLattice(new_state)
    # Set lattice tensor
    for l1, l2 in new_state.sites():
        new_state[l1, l2] = temporary_lattice[l1][l2]
    return new_state, old_state
