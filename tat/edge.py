"""
This file contains the definition of tensor edge.
"""

from __future__ import annotations
import functools
import operator
import typing
import torch
from . import _utility


class Edge:
    """
    The edge type of tensor.
    """

    __slots__ = "_fermion", "_dtypes", "_symmetry", "_dimension", "_arrow", "_parity"

    @property
    def fermion(self: Edge) -> list[bool]:
        """
        A list records whether every sub symmetry is fermionic. Its length is the number of sub symmetry.
        """
        return self._fermion

    @property
    def dtypes(self: Edge) -> list[torch.dtype]:
        """
        A list records the basic dtype of every sub symmetry. Its length is the number of sub symmetry.
        """
        return self._dtypes

    @property
    def symmetry(self: Edge) -> list[torch.Tensor]:
        """
        A list containing all symmetry of this edge. Its length is the number of sub symmetry. Every element of it is a
        sub symmetry.
        """
        return self._symmetry

    @property
    def dimension(self: Edge) -> int:
        """
        The dimension of this edge.
        """
        return self._dimension

    @property
    def arrow(self: Edge) -> bool:
        """
        The arrow of this edge.
        """
        return self._arrow

    @property
    def parity(self: Edge) -> torch.Tensor:
        """
        The parity of this edge.
        """
        return self._parity

    def __init__(
        self: Edge,
        *,
        fermion: typing.Optional[list[bool]] = None,
        dtypes: typing.Optional[list[torch.dtype]] = None,
        symmetry: typing.Optional[list[torch.Tensor]] = None,
        dimension: typing.Optional[int] = None,
        arrow: typing.Optional[bool] = None,
        # The following argument is not public
        parity: typing.Optional[torch.Tensor] = None,
    ) -> None:
        """
        Create an edge with essential information.

        Examples:
        - Edge(dimension=5)
        - Edge(symmetry=[torch.tensor([False, False, True, True])])
        - Edge(fermion=[False, True], symmetry=[torch.tensor([False, True]), torch.tensor([False, True])], arrow=True)

        Parameters
        ----------
        fermion : list[bool], optional
            Whether each sub symmetry is fermionic symmetry, its length should be the same to symmetry. But it could be
            left empty, if so, a total bosonic edge will be created.
        dtypes : list[torch.dtype], optional
            The basic dtype to identify each sub symmetry, its length should be the same to symmetry, and it is nothing
            but the dtypes of each tensor in the symmetry. It could be left empty, if so, it will be derived from
            symmetry.
        symmetry : list[torch.Tensor], optional
            The symmetry information of every sub symmetry, each of sub symmetry should be a one dimensional tensor with
            the same length dimension, and their dtype should be integral type, aka, int or bool.
        dimension : int, optional
            The dimension of the edge, if not specified, dimension will be detected from symmetry.
        arrow : bool, optional
            The arrow direction of the edge, it is essential for fermionic edge, aka, an edge with fermionic sub
            symmetry.
        """
        # pylint: disable=too-many-arguments

        # Symmetry could be left empty to create no symmetry edge
        if symmetry is None:
            symmetry = []

        # Fermion could be empty if it is total bosonic edge
        if fermion is None:
            fermion = [False for _ in symmetry]

        # Dtypes could be empty and derived from symmetry
        if dtypes is None:
            dtypes = [sub_symmetry.dtype for sub_symmetry in symmetry]
        # Check dtype is compatible with symmetry
        assert all(sub_symmetry.dtype is sub_dtype for sub_symmetry, sub_dtype in zip(symmetry, dtypes))
        # Check dtype is valid, aka, bool or int
        assert all(not (sub_symmetry.is_floating_point() or sub_symmetry.is_complex()) for sub_symmetry in symmetry)

        # The fermion, dtypes and symmetry information should have the same length
        assert len(fermion) == len(dtypes) == len(symmetry)

        # If dimension not set, get dimension from symmetry
        if dimension is None:
            dimension = len(symmetry[0])
        # Check if the dimensions of different sub_symmetry mismatch
        assert all(sub_symmetry.size() == (dimension,) for sub_symmetry in symmetry)

        if arrow is None:
            # Arrow not set, it should be bosonic edge.
            arrow = False
            assert not any(fermion)

        self._fermion: list[bool] = fermion
        self._dtypes: list[torch.dtype] = dtypes
        self._symmetry: list[torch.Tensor] = symmetry
        self._dimension: int = dimension
        self._arrow: bool = arrow

        if parity is None:
            parity = self._generate_parity()
        self._parity: torch.Tensor = parity
        assert self.parity.size() == (self.dimension,)
        assert self.parity.dtype is torch.bool

    def _generate_parity(self: Edge) -> torch.Tensor:
        return functools.reduce(
            # Reduce sub parity for all sub symmetry by logical xor
            torch.logical_xor,
            (
                # The parity of sub symmetry
                _utility.parity(sub_symmetry)
                # Loop all sub symmetry
                for sub_symmetry, sub_fermion in zip(self.symmetry, self.fermion)
                # But only reduce if it is fermion sub symmetry
                if sub_fermion),
            # Reduce with start as tensor filled with False
            torch.zeros(self.dimension, dtype=torch.bool),
        )

    def conjugate(self: Edge) -> Edge:
        """
        Get the conjugated edge.

        Returns
        -------
        Edge
            The conjugated edge.
        """
        # The only two difference of conjugated edge is symmetry and arrow
        return Edge(
            fermion=self.fermion,
            dtypes=self.dtypes,
            symmetry=[
                _utility.neg_symmetry(sub_symmetry)  # bool -> same, int -> neg
                for sub_symmetry in self.symmetry
            ],
            dimension=self.dimension,
            arrow=not self.arrow,
            parity=self.parity,
        )

    def __eq__(self: Edge, other: typing.Any) -> bool:
        if not isinstance(other, Edge):
            # pylint: disable=no-else-return
            if torch.jit.is_scripting():
                return False
            else:
                return NotImplemented
        return (
            # Compare int dimension and bool arrow first since they are fast to compare
            self.dimension == other.dimension and
            # But even if arrow are different, if it is bosonic edge, it is also OK
            (self.arrow == other.arrow or not any(self.fermion)) and
            # Then the list of bool are compared
            self.fermion == other.fermion and
            # Then the list of dtypes are compared
            self.dtypes == other.dtypes and
            # All of symmetries are compared at last, since it is biggest
            all(
                torch.equal(self_sub_symmetry, other_sub_symmetry)
                for self_sub_symmetry, other_sub_symmetry in zip(self.symmetry, other.symmetry)))

    def __str__(self: Edge) -> str:
        # pylint: disable=no-else-return
        if any(self.fermion):
            # Fermionic edge
            fermion = ','.join(str(sub_fermion) for sub_fermion in self.fermion)
            symmetry = ','.join(
                f"[{','.join(str(sub_sym.item()) for sub_sym in sub_symmetry)}]" for sub_symmetry in self.symmetry)
            return f"(dimension={self.dimension}, arrow={self.arrow}, fermion=({fermion}), symmetry=({symmetry}))"
        elif self.fermion:
            # Bosonic edge
            symmetry = ','.join(
                f"[{','.join(str(sub_sym.item()) for sub_sym in sub_symmetry)}]" for sub_symmetry in self.symmetry)
            return f"(dimension={self.dimension}, symmetry=({symmetry}))"
        else:
            # Trivial edge
            return f"(dimension={self.dimension})"

    def __repr__(self: Edge) -> str:
        return f"Edge{self.__str__()}"

    @staticmethod
    def merge_edges(
        edges: list[Edge],
        *,
        fermion: typing.Optional[list[bool]] = None,
        dtypes: typing.Optional[list[torch.dtype]] = None,
        arrow: typing.Optional[bool] = None,
    ) -> Edge:
        """
        Merge several edges into one edge.

        Parameters
        ----------
        edges : list[Edge]
            The edges to be merged.
        fermion : list[bool], optional
            Whether each sub symmetry is fermionic, it could be left empty to derive from edges
        dtypes : list[torch.dtype], optional
            The base type of sub symmetry, it could be left empty to derive from edges
        arrow : bool, optional
            The arrow of all the edges, it is useful if edges is empty.

        Returns
        -------
        Edge
            The result edge merged by edges.
        """
        # If fermion not set, get it from edges
        if fermion is None:
            fermion = edges[0].fermion
        # All edge should share the same fermion
        assert all(fermion == edge.fermion for edge in edges)
        # If dtypes not set, get it from edges
        if dtypes is None:
            dtypes = edges[0].dtypes
        # All edge should share the same dtypes
        assert all(dtypes == edge.dtypes for edge in edges)
        # If arrow set, check it directly, if not set, set to False or get from edges
        if arrow is None:
            if any(fermion):
                # It is fermionic edge.
                arrow = edges[0].arrow
            else:
                # It is bosonic edge, set to False directly since it is useless.
                arrow = False
        # All edge should share the same arrow for fermionic edge
        assert (not any(fermion)) or all(arrow == edge.arrow for edge in edges)

        rank = len(edges)
        # Merge edge
        dimension = functools.reduce(operator.mul, (edge.dimension for edge in edges), 1)
        symmetry = [
            # Every merged sub symmetry is calculated by reduce and flatten
            functools.reduce(
                # The reduce operator depend on the dtype of this sub symmetry
                _utility.add_symmetry,
                (
                    # The sub symmetry of every edge will be reshape to be reduced.
                    _utility.unsqueeze(edge.symmetry[sub_symmetry_index], current_index, rank)
                    # The sub symmetry of every edge is reduced one by one
                    for current_index, edge in enumerate(edges)),
                # Reduce from a rank-0 tensor
                torch.zeros([], dtype=sub_symmetry_dtype),
            ).reshape([-1])
            # Merge every sub symmetry one by one
            for sub_symmetry_index, sub_symmetry_dtype in enumerate(dtypes)
        ]

        # parity not set here since it need recalculation
        return Edge(
            fermion=fermion,
            dtypes=dtypes,
            symmetry=symmetry,
            dimension=dimension,
            arrow=arrow,
        )
