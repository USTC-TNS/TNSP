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

from .abstract_state import AbstractState


class AbstractLatticeVirtualBond:
    """
    Virtual bond handler for abstract lattice.
    """

    __slots__ = ["_owner"]

    def __init__(self, owner):
        """
        Create virtual bond handler.

        Parameters
        ----------
        owner : AbstractState
            The owner of this handler.
        """
        self._owner = owner

    def __getitem__(self, where):
        """
        Get the virtual bond edge.

        Parameters
        ----------
        where : tuple[int, int, str]
            The coordinate and the direction to find the bond.

        Returns
        -------
        Edge
            The virtual bond edge.
        """
        l1, l2, direction = where
        return self._owner._virtual_bond[l1, l2][direction]

    def __setitem__(self, where, value):
        """
        Set the virtual bond edge.

        Parameters
        ----------
        where : tuple[int, int, str] | str
            The coordinate and the direction to find the bond, if it is str, set this direction for every site to the
            same edge.
        value : ?Edge
            The virtual bond edge.
        """
        if isinstance(where, str):
            direction = where
            for l1 in range(self._owner.L1):
                for l2 in range(self._owner.L2):
                    self._owner._set_virtual_bond((l1, l2, direction), value)
        else:
            l1, l2, direction = where
            self._owner._set_virtual_bond((l1, l2, direction), value)


class AbstractLattice(AbstractState):
    """
    The abstract lattice.
    """

    __slots__ = ["_virtual_bond"]

    def __init__(self, abstract):
        """
        Create abstract lattice from a given abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create abstract lattice.
        """
        super()._init_by_copy(abstract)

        self._virtual_bond = {(l1, l2): self._default_bonds(l1, l2) for l1 in range(self.L1) for l2 in range(self.L2)}

    def _init_by_copy(self, other):
        """
        Copy constructor of abstract lattice.

        Parameters
        ----------
        other : AbstractLattice
            Another abstract lattice.
        """
        super()._init_by_copy(other)

        self._virtual_bond = {
            l1l2: {direction: edge for direction, edge in m.items()} for l1l2, m in other._virtual_bond.items()
        }

    def _construct_tensor(self, l1, l2):
        """
        Construct tensor for this abstract lattice, only called for derived type of abstract lattice.

        Parameters
        ----------
        l1 : int
            The height index of the site.
        l2 : int
            The weight index of the site.

        Returns
        -------
        Tensor
            The site tensor created by the edge recorded in abstract lattice.
        """
        physics_edges = self.physics_edges[l1, l2]
        names = [f"P{orbit}" for orbit, edge in physics_edges.items()]
        edges = [edge for orbit, edge in physics_edges.items()]
        for direction, edge in self._virtual_bond[l1, l2].items():
            if edge is not None:
                names.append(direction)
                edges.append(edge)
        return self.Tensor(names, edges).randn()

    def _default_bonds(self, l1, l2):
        """
        Get the default bond for each site.

        Parameters
        ----------
        l1 : int
            The height index of the site.
        l2 : int
            The weight index of the site.

        Returns
        -------
        dict[str, Edge]
            The map from direction to edge.
        """
        result = {}
        if l1 == l2 == 0:
            result["T"] = self._total_symmetry_edge
        if l1 != 0:
            result["U"] = None
        if l1 != self.L1 - 1:
            result["D"] = None
        if l2 != 0:
            result["L"] = None
        if l2 != self.L2 - 1:
            result["R"] = None
        return result

    @property
    def virtual_bond(self):
        """
        Get the virtual bond handler of this abstract lattice.

        Returns
        -------
            The virtual bond handler.
        """
        return AbstractLatticeVirtualBond(self)

    def _set_virtual_bond_single_side(self, where, edge):
        """
        Set the virtual bond edge for single side only, used by _set_virtual_bond.

        Parameters
        ----------
        where : tuple[int, int, str]
            The coordinate and the direction to find the bond. if the bond is missing, nothing will happen.
        value : Edge
            The virtual bond edge.
        """
        l1, l2, direction = where
        l1l2 = l1, l2
        if l1l2 in self._virtual_bond:
            site = self._virtual_bond[l1l2]
            if direction in site:
                site[direction] = edge

    def _set_virtual_bond(self, where, edge):
        """
        Set the virtual bond edge.

        Parameters
        ----------
        where : tuple[int, int, str]
            The coordinate and the direction to find the bond.
        value : ?Edge
            The virtual bond edge.
        """
        l1, l2, direction = where
        edge = self._construct_edge(edge)
        self._set_virtual_bond_single_side((l1, l2, direction), edge)
        if direction == "L":
            l2 -= 1
            direction = "R"
        elif direction == "R":
            l2 += 1
            direction = "L"
        elif direction == "U":
            l1 -= 1
            direction = "D"
        elif direction == "D":
            l1 += 1
            direction = "U"
        else:
            raise ValueError("Invalid direction when setting virtual bond")
        self._set_virtual_bond_single_side((l1, l2, direction), edge.conjugated())
