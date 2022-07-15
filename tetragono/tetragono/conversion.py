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

from .exact_state import ExactState
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice


def simple_update_lattice_to_sampling_lattice(state):
    """
    Convert SimpleUpdateLattice to ExactState

    Parameters
    ----------
    state : SimpleUpdateLattice

    Returns
    -------
    ExactState
    """
    if not isinstance(state, SimpleUpdateLattice):
        raise ValueError("Conversion input type mismatch")
    result = SamplingLattice(state)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            this = state[l1, l2]
            this = state._try_multiple(this, l1, l2, "L", division=True, square_root=True)
            this = state._try_multiple(this, l1, l2, "U", division=True, square_root=True)
            this = state._try_multiple(this, l1, l2, "R", division=True, square_root=True)
            this = state._try_multiple(this, l1, l2, "D", division=True, square_root=True)
            result[l1, l2] = this
    return result


def simple_update_lattice_to_exact_state(state):
    """
    Convert SimpleUpdateLattice to ExactState

    Parameters
    ----------
    state : SimpleUpdateLattice

    Returns
    -------
    ExactState
    """
    if not isinstance(state, SimpleUpdateLattice):
        raise ValueError("Conversion input type mismatch")
    result = ExactState(state)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            rename_map = {f"P{orbit}": f"P_{l1}_{l2}_{orbit}" for orbit, edge in state.physics_edges[l1, l2].items()}
            if l1 != state.L1 - 1:
                rename_map["D"] = f"D_{l2}"
            this = state[l1, l2].edge_rename(rename_map)
            this = state._try_multiple(this, l1, l2, "L", division=True)
            this = state._try_multiple(this, l1, l2, "U", division=True)
            if l1 == l2 == 0:
                result.vector = this
            else:
                contract_pair = set()
                if l2 != 0:
                    contract_pair.add(("R", "L"))
                if l1 != 0:
                    contract_pair.add((f"D_{l2}", "U"))
                result.vector = result.vector.contract(this, contract_pair)
    return result


def sampling_lattice_to_exact_state(state):
    """
    Convert SamplingLattice to ExactState

    Parameters
    ----------
    state : SamplingLattice

    Returns
    -------
    ExactState
    """
    if not isinstance(state, SamplingLattice):
        raise ValueError("Conversion input type mismatch")
    result = ExactState(state)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            rename_map = {f"P{orbit}": f"P_{l1}_{l2}_{orbit}" for orbit, edge in state.physics_edges[l1, l2].items()}
            if l1 != state.L1 - 1:
                rename_map["D"] = f"D_{l2}"
            this = state[l1, l2].edge_rename(rename_map)
            if l1 == l2 == 0:
                result.vector = this
            else:
                contract_pair = set()
                if l2 != 0:
                    contract_pair.add(("R", "L"))
                if l1 != 0:
                    contract_pair.add((f"D_{l2}", "U"))
                result.vector = result.vector.contract(this, contract_pair)
    return result
