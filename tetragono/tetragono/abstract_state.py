#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


class AbstractStatePhysicsEdge:
    """
    Physics edge handler for abstract state.
    """

    __slots__ = ["owner"]

    def __init__(self, owner):
        """
        Create a physics edge handler.

        Parameters
        ----------
        owner : AbstractState
            The owner of this handler.
        """
        self.owner: AbstractState = owner

    def __getitem__(self, l1l2o):
        """
        Get the physics edge from abstract state.

        Get a specific orbit edge from state.physics_edge[l1, l2, orbit].
        Get all orbit of a site from state.physics_edge[l1, l2] as a dict from orbit index to edge.

        Parameters
        ----------
        l1l2o : tuple[int, int, int] | tuple[int, int]
            The position of physics edge, if it is tuple[int, int], it is coordinate of the site, if the third int is
            given, the third int is the orbit index.

        Returns
        -------
        Edge | dict[int, Edge]
            If the orbit index is given, return the Edge of this orbit, if orbit is not specified, return a dict mapping
            orbit index to Edge.
        """
        if len(l1l2o) == 3:
            l1, l2, orbit = l1l2o
            return self.owner._physics_edges[l1][l2][orbit]
        elif len(l1l2o) == 2:
            l1, l2 = l1l2o
            return self.owner._physics_edges[l1][l2]
        else:
            raise ValueError("Invalid getitem argument for physics edge handler")

    def __iter__(self):
        """
        Iterate on physics edges.
        """
        for l1, l2 in self.owner.sites():
            for orbit, edge in self.owner._physics_edges[l1][l2].items():
                yield (l1, l2, orbit), edge

    def __setitem__(self, l1l2o, edge):
        """
        Set the physics edge for abstract state.

        Use state.physics_edge[...] = xxx to set all site as single oribt.
        Use state.physics_edge[l1, l2] = xxx to set a specific site to a single orbit site.
        Use state.physics_edge[l1, l2, orbit] = xxx to set the edge of specific orbit in some site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int] | tuple[int, int] | type(...)
            The coordinate and orbit index of the physics edge to be set, if l1l2o is `...`, set single orbit for all
            site and set edges of all sites to the same edge.
        edge : ?Edge
            An edge or something that can be used to construct an edge.
        """
        self.owner._site_number = None
        edge = self.owner._construct_physics_edge(edge)
        if l1l2o is ...:
            self.owner._physics_edges = [[{0: edge} for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]
        elif len(l1l2o) == 3:
            l1, l2, orbit = l1l2o
            if orbit < 0:
                raise ValueError("Orbit index should be non-negative")
            self.owner._physics_edges[l1][l2][orbit] = edge
            # Once the edge has been set, resort the dict of this site
            self.owner._physics_edges[l1][l2] = dict(sorted(self.owner._physics_edges[l1][l2].items()))
        elif len(l1l2o) == 2:
            l1, l2 = l1l2o
            self.owner._physics_edges[l1][l2] = {0: edge}
        else:
            raise ValueError("Invalid setitem argument for physics edge handler")


class AbstractStateHamiltonian:
    """
    Hamiltonian handler for abstract state.
    """

    __slots__ = ["owner"]

    def __init__(self, owner):
        """
        Create a hamiltonian handler.

        Parameters
        ----------
        owner : AbstractState
            The owner of this handler.
        """
        self.owner: AbstractState = owner

    def __getitem__(self, points):
        """
        Get a hamitlonian for several points.

        Parameters
        ----------
        points : tuple[tuple[int, int, int] | tuple[int, int], ...]
            List of points which the hamiltonian applies on, every point is a tuple[int, int, int], the first two int
            is coordinate and the third is orbit index. The orbit index could be eliminate for the first orbit.

        Returns
        -------
        Tensor
            The hamiltonian tensor
        """
        points = tuple(point if len(point) == 3 else (point[0], point[1], 0) for point in points)
        return self.owner._hamiltonians[points]

    def __iter__(self):
        """
        Iterate on the hamiltonians
        """
        return iter(sorted(self.owner._hamiltonians.items()))

    def __contains__(self, key):
        """
        Check the existence of hamiltonian term on specified points.
        """
        return key in self.owner._hamiltonians

    def __len__(self):
        """
        Get the number of hamiltonian terms.
        """
        return len(self.owner._hamiltonians)

    def __setitem__(self, arg, tensor):
        """
        Set a hamiltonian for several points.

        Parameters
        ----------
        arg : tuple[tuple[int, int, int] | tuple[int, int], ...] | str
            If arg is tuple, it is list of points which the hamiltonian applies on, every point is a
            tuple[int, int, int] for full specification of a orbit or a tuple[int, int] to specify the first orbit of a
            site. If arg is str, it is used to set some common used kinds of hamiltonian.
        tensor : Tensor
             The hamiltonian tensor.
        """
        if isinstance(arg, str):
            if arg == "single_site":
                # Set hamiltonian to all first orbit of every site
                for l1, l2 in self.owner.sites():
                    self.owner._set_hamiltonian(((l1, l2, 0),), tensor)
            elif arg == "vertical_bond":
                # Set hamiltonian to all vertical bond connecting first orbits
                for l1, l2 in self.owner.sites():
                    if l1 == self.owner.L1 - 1:
                        continue
                    self.owner._set_hamiltonian(((l1, l2, 0), (l1 + 1, l2, 0)), tensor)
            elif arg == "horizontal_bond":
                # Set hamiltonian to all horinzontal bond connecting first orbits
                for l1, l2 in self.owner.sites():
                    if l2 == self.owner.L2 - 1:
                        continue
                    self.owner._set_hamiltonian(((l1, l2, 0), (l1, l2 + 1, 0)), tensor)
            else:
                raise ValueError("Unknown kind of hamiltonian")
        else:
            points = tuple(point if len(point) == 3 else (point[0], point[1], 0) for point in arg)
            self.owner._set_hamiltonian(points, tensor)


class AbstractState:
    """
    Abstract state, which is used to construct other type of state.
    """

    __slots__ = [
        "Tensor", "L1", "L2", "_physics_edges", "_hamiltonians", "_total_symmetry", "_site_number", "data_version",
        "attribute"
    ]

    def sites(self):
        """
        Iterate on site.

        Yields
        ------
        int, int
            The site of the system.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                yield l1, l2

    @property
    def Edge(self):
        """
        Get the edge type of this abstract state.

        Returns
        -------
        type
            The edge type of this abstract state.
        """
        return self.Tensor.model.Edge

    @property
    def Symmetry(self):
        """
        Get the symmetry type of this abstract state.

        Returns
        -------
        type
            The symmetry type of this abstract state.
        """
        return self.Tensor.model.Symmetry

    def _v2_to_v3_rename(self, state):
        """
        Update the data from version 2 to version 3.

        From version 2 to version 3, several member renamed.
        """
        state["L1"] = state["_L1"]
        state["L2"] = state["_L2"]
        state["Tensor"] = state["_Tensor"]
        del state["_L1"]
        del state["_L2"]
        del state["_Tensor"]

    def _v5_to_v6_attribute(self, state):
        """
        Update the data from version 5 to version 6.

        From version 5 to version 6, a member called attribute is added, which is a dict.
        """
        state["attribute"] = {}

    def __init__(self, Tensor, L1, L2):
        """
        Create an abstract state.

        Parameters
        ----------
        Tensor : type
            The tensor type of this abstract state.
        L1 : int
            The square system size of this abstract state.
        L2 : int
            The square system size of this abstract state.
        """

        # The tensor type of this abstract state
        self.Tensor = Tensor
        # The system size.
        self.L1 = L1
        self.L2 = L2

        # Data storage for physics edge, access it by state.physics_edge instead
        self._physics_edges = [[{} for l2 in range(self.L2)] for l1 in range(self.L1)]
        # Data storage for hamiltonians, access it by state.hamiltonians instead
        self._hamiltonians = {}
        # The total symmetry of the whole state, access it by state.total_symmetry
        self._total_symmetry = self.Symmetry()
        # The total site number of the whole state, access it by state.site_number
        self._site_number = None

        # Customed attribute used by user
        self.attribute = {}
        # This attribute is also used for experiment feature.
        # Current experiment feature list is
        # - alpha: float
        #       Use p^alpha instead on p as possibility in sampling method.
        # - smalldiag: tuple[float, float, float]
        #       Smaller diagonal possibility item for density matrix in sampling method.
        #       Three parts of `smalldiag` are minimum, maximum, rate.
        #       This attribute requires user to maintain attribute "beta" manually.
        #       The coefficient is beta*rate, but clamped between minimum and maximum.

        self.data_version = 6

    def _init_by_copy(self, other):
        """
        Copy constructor of abstract state.

        Parameters
        ----------
        other : AbstractState
            Another abstract state.
        """
        self.Tensor = other.Tensor
        self.L1 = other.L1
        self.L2 = other.L2
        self._physics_edges = [[other._physics_edges[l1][l2].copy() for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._hamiltonians = other._hamiltonians.copy()
        self._total_symmetry = other._total_symmetry
        self._site_number = other._site_number
        self.attribute = other.attribute
        self.data_version = other.data_version

    def _construct_symmetry(self, value):
        """
        Construct symmetry from something that can be used to construct an symmetry.

        Parameters
        ----------
        value : ?Symmetry
            Symmetry or something that can be used to construct a symmetry.

        Returns
        -------
        Symmetry
            The result symmetry object.
        """
        return self.Symmetry(value)

    @property
    def total_symmetry(self):
        """
        Get the total symmetry of this abstract state

        Returns
        -------
        Symmetry
            The total symmetry of this abstract state
        """
        return self._total_symmetry

    @total_symmetry.setter
    def total_symmetry(self, value):
        """
        Set the total symmetry of this abstract state

        Parameters
        ----------
        value : ?Symmetry
            The total symmetry of this abstract state
        """
        self._total_symmetry = self._construct_symmetry(value)

    @property
    def _total_symmetry_edge(self):
        """
        Get the virtual edge for containing total symmetry.

        Returns
        -------
        Edge
            The result virtual edge.
        """
        return self.Edge([(-self._total_symmetry, 1)], False)

    def _construct_edge(self, value):
        """
        Construct edge from something that can be used to construct an edge.

        Parameters
        ----------
        value : ?Edge
            Edge or something that can be used to construct an edge.

        Returns
        -------
        Edge
            The result edge object.
        """
        return self.Edge(value)

    def _construct_physics_edge(self, edge):
        """
        Construct physics edge from something that can be used to construct an edge.

        Parameters
        ----------
        value : ?Edge
            Edge or something that can be used to construct an edge.

        Returns
        -------
        Edge
            The result edge object used for physics edge.
        """
        result = self._construct_edge(edge)
        if result.arrow is not False:
            raise ValueError("Edge arrow of physics bond should be False")
        return result

    @property
    def physics_edges(self):
        """
        Get the physics edge handler for this abstract state.

        Returns
        -------
        AbstractStatePhysicsEdge
            The physics edge handler.
        """
        return AbstractStatePhysicsEdge(self)

    @property
    def hamiltonians(self):
        """
        Get the hamiltonian handler for this abstract state.

        Returns
        -------
        AbstractStateHamiltonian
            The hamiltonian handler.
        """
        return AbstractStateHamiltonian(self)

    def _set_hamiltonian(self, points, tensor):
        """
        Set a hamiltonian for several points

        Parameters
        ----------
        points : tuple[tuple[int, int, int], ...]
            List of points which the hamiltonian applies on, every point is a tuple[int, int, int], the first two int is
            coordinate and the third is orbit index.
        tensor : Tensor
             The hamiltonian tensor.
        """
        body = len(points)
        if not isinstance(tensor, self.Tensor):
            raise TypeError("Wrong hamiltonian type")
        if {f"{i}" for i in tensor.names} != {f"{i}{j}" for i in ["I", "O"] for j in range(body)}:
            raise ValueError("Wrong hamiltonian name")
        for i in range(body):
            edge_out = tensor.edge_by_name(f"O{i}")
            edge_in = tensor.edge_by_name(f"I{i}")
            if edge_out != self.physics_edges[points[i]]:
                raise ValueError("Wrong hamiltonian edge")
            if edge_out.conjugate() != edge_in:
                raise ValueError("Wrong hamiltonian edge")

        if tensor.norm_max() != 0:
            self._hamiltonians[points] = tensor

    @property
    def site_number(self):
        """
        Get the total site number of this abstract state.

        Returns
        -------
        int
            The total site number.
        """
        if self._site_number is None:
            self._site_number = len(list(self.physics_edges))
        return self._site_number

    def numpy_hamiltonian(self):
        """
        Get the numpy array as the hamiltonian. It is dangerous to call this function on a large model.

        Returns
        -------
        np.ndarray
            The hamiltonian in numpy array format.
        """
        edge_order = [l1l2o for l1l2o, edge in self.physics_edges]
        result = None
        for positions, hamiltonian in self.hamiltonians:
            body = len(positions)
            this_term = hamiltonian.edge_rename({f"I{i}": f"I{edge_order.index(positions[i])}" for i in range(body)} |
                                                {f"O{i}": f"O{edge_order.index(positions[i])}" for i in range(body)})
            for l1l2o, edge in self.physics_edges:
                if l1l2o not in positions:
                    index = edge_order.index(l1l2o)
                    identity = (
                        self.Tensor([f"O{index}", f"I{index}"], [edge, edge.conjugate()])  #
                        .identity_({(f"O{index}", f"I{index}")}))
                    this_term = this_term.contract(identity, set())
            # Here, apply merge sign to "I" but not to "O"
            # In fact, it is also OK if apply merge sign to "O" instead of "I"
            this_array = this_term.merge_edge(
                {
                    "O": [f"O{i}" for i in range(self.site_number)],
                    "I": [f"I{i}" for i in range(self.site_number)]
                }, True, {"O"}).const_blocks[[("I", -self.total_symmetry), ("O", self.total_symmetry)]]
            if result is None:
                result = this_array
            else:
                result += this_array
        return result
