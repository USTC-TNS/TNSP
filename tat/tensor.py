"""
This file defined the core tensor type for tat package.
"""

from __future__ import annotations
import typing
import operator
import functools
from multimethod import multimethod
import torch
from . import _utility
from ._qr import givens_qr, householder_qr  # pylint: disable=unused-import
from ._svd import svd as manual_svd  # pylint: disable=unused-import
from .edge import Edge

# pylint: disable=too-many-public-methods
# pylint: disable=too-many-lines


class Tensor:
    """
    The main tensor type, which wraps pytorch tensor and provides edge names and Fermionic functions.
    """

    __slots__ = "_fermion", "_dtypes", "_names", "_edges", "_data", "_mask"

    def __str__(self: Tensor) -> str:
        return f"(names={self.names}, edges={self.edges}, data={self.data})"

    def __repr__(self: Tensor) -> str:
        return f"Tensor(names={self.names}, edges={self.edges})"

    @property
    def fermion(self: Tensor) -> list[bool]:
        """
        A list records whether every sub symmetry is fermionic. Its length is the number of sub symmetry.
        """
        return self._fermion

    @property
    def dtypes(self: Tensor) -> list[torch.dtype]:
        """
        A list records the basic dtype of every sub symmetry. Its length is the number of sub symmetry.
        """
        return self._dtypes

    @property
    def names(self: Tensor) -> list[str]:
        """
        The edge names of this tensor.
        """
        return self._names

    @property
    def edges(self: Tensor) -> list[Edge]:
        """
        The edges information of this tensor.
        """
        return self._edges

    @property
    def data(self: Tensor) -> torch.Tensor:
        """
        The content data of this tensor.
        """
        return self._data

    @property
    def mask(self: Tensor) -> torch.Tensor:
        """
        The content data mask of this tensor.
        """
        return self._mask

    @property
    def rank(self: Tensor) -> int:
        """
        The rank of this tensor.
        """
        return len(self._names)

    @property
    def dtype(self: Tensor) -> torch.dtype:
        """
        The data type of the content in this tensor.
        """
        return self.data.dtype

    @property
    def btype(self: Tensor) -> str:
        """
        The data type of the content in this tensor, represented in BLAS/LAPACK convention.
        """
        if self.dtype is torch.float32:
            return 'S'
        if self.dtype is torch.float64:
            return 'D'
        if self.dtype is torch.complex64:
            return 'C'
        if self.dtype is torch.complex128:
            return 'Z'
        return '?'

    @property
    def is_complex(self: Tensor) -> bool:
        """
        Whether it is a complex tensor
        """
        return self.dtype.is_complex

    @property
    def is_real(self: Tensor) -> bool:
        """
        Whether it is a real tensor
        """
        return self.dtype.is_floating_point

    def edge_by_name(self: Tensor, name: str) -> Edge:
        """
        Get edge by the edge name of this tensor.

        Parameters
        ----------
        name : str
            The given edge name.

        Returns
        -------
        Edge
            The edge with the given edge name.
        """
        assert name in self.names
        return self.edges[self.names.index(name)]

    def _arithmetic_operator(self: Tensor, other: object, operate: typing.Callable) -> Tensor:
        new_data: torch.Tensor
        if isinstance(other, Tensor):
            # If it is tensor, check same shape and transpose before calculating.
            assert self.same_shape_with(other)
            new_data = operate(self.data, other.transpose(self.names).data)
            if operate is torch.div:
                # In div, it may generate nan
                new_data = torch.where(self.mask, new_data, torch.zeros([], dtype=self.dtype))
        else:
            # Otherwise treat other as a scalar, mask should be applied later.
            new_data = operate(self.data, other)
            new_data = torch.where(self.mask, new_data, torch.zeros([], dtype=self.dtype))
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=new_data,
            mask=self.mask,
        )

    def __add__(self: Tensor, other: object) -> Tensor:
        return self._arithmetic_operator(other, torch.add)

    def __sub__(self: Tensor, other: object) -> Tensor:
        return self._arithmetic_operator(other, torch.sub)

    def __mul__(self: Tensor, other: object) -> Tensor:
        return self._arithmetic_operator(other, torch.mul)

    def __truediv__(self: Tensor, other: object) -> Tensor:
        return self._arithmetic_operator(other, torch.div)

    def _right_arithmetic_operator(self: Tensor, other: object, operate: typing.Callable) -> Tensor:
        new_data: torch.Tensor
        if isinstance(other, Tensor):
            # If it is tensor, check same shape and transpose before calculating.
            assert self.same_shape_with(other)
            new_data = operate(other.transpose(self.names).data, self.data)
            if operate is torch.div:
                # In div, it may generate nan
                new_data = torch.where(self.mask, new_data, torch.zeros([], dtype=self.dtype))
        else:
            # Otherwise treat other as a scalar, mask should be applied later.
            new_data = operate(other, self.data)
            new_data = torch.where(self.mask, new_data, torch.zeros([], dtype=self.dtype))
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=new_data,
            mask=self.mask,
        )

    def __radd__(self: Tensor, other: object) -> Tensor:
        return self._right_arithmetic_operator(other, torch.add)

    def __rsub__(self: Tensor, other: object) -> Tensor:
        return self._right_arithmetic_operator(other, torch.sub)

    def __rmul__(self: Tensor, other: object) -> Tensor:
        return self._right_arithmetic_operator(other, torch.mul)

    def __rtruediv__(self: Tensor, other: object) -> Tensor:
        return self._right_arithmetic_operator(other, torch.div)

    def _inplace_arithmetic_operator(self: Tensor, other: object, operate: typing.Callable) -> Tensor:
        if isinstance(other, Tensor):
            # If it is tensor, check same shape and transpose before calculating.
            assert self.same_shape_with(other)
            operate(self.data, other.transpose(self.names).data, out=self.data)
            if operate is torch.div:
                # In div, it may generate nan
                torch.where(self.mask, self.data, torch.zeros([], dtype=self.dtype), out=self.data)
        else:
            # Otherwise treat other as a scalar, mask should be applied later.
            operate(self.data, other, out=self.data)
            torch.where(self.mask, self.data, torch.zeros([], dtype=self.dtype), out=self.data)
        return self

    def __iadd__(self: Tensor, other: object) -> Tensor:
        return self._inplace_arithmetic_operator(other, torch.add)

    def __isub__(self: Tensor, other: object) -> Tensor:
        return self._inplace_arithmetic_operator(other, torch.sub)

    def __imul__(self: Tensor, other: object) -> Tensor:
        return self._inplace_arithmetic_operator(other, torch.mul)

    def __itruediv__(self: Tensor, other: object) -> Tensor:
        return self._inplace_arithmetic_operator(other, torch.div)

    def __pos__(self: Tensor) -> Tensor:
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=+self.data,
            mask=self.mask,
        )

    def __neg__(self: Tensor) -> Tensor:
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=-self.data,
            mask=self.mask,
        )

    def __float__(self: Tensor) -> float:
        return float(self.data)

    def __complex__(self: Tensor) -> complex:
        return complex(self.data)

    def norm(self: Tensor, order: typing.Any) -> float:
        """
        Get the norm of the tensor, this function will flatten tensor first before calculate norm.

        Parameters
        ----------
        order
            The order of norm.

        Returns
        -------
        float
            The norm of the tensor.
        """
        return torch.linalg.vector_norm(self.data, ord=order)

    def norm_max(self: Tensor) -> float:
        "max norm"
        return self.norm(+torch.inf)

    def norm_min(self: Tensor) -> float:
        "min norm"
        return self.norm(-torch.inf)

    def norm_num(self: Tensor) -> float:
        "0-norm"
        return self.norm(0)

    def norm_sum(self: Tensor) -> float:
        "1-norm"
        return self.norm(1)

    def norm_2(self: Tensor) -> float:
        "2-norm"
        return self.norm(2)

    def copy(self: Tensor) -> Tensor:
        """
        Get a copy of this tensor

        Returns
        -------
        Tensor
            The copy of this tensor
        """
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=torch.clone(self.data, memory_format=torch.contiguous_format),
            mask=self.mask,
        )

    def __copy__(self: Tensor) -> Tensor:
        return self.copy()

    def __deepcopy__(self: Tensor, _: typing.Any = None) -> Tensor:
        return self.copy()

    def same_shape(self: Tensor) -> Tensor:
        """
        Get a tensor with same shape to this tensor

        Returns
        -------
        Tensor
            A new tensor with the same shape to this tensor
        """
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=torch.zeros_like(self.data),
            mask=self.mask,
        )

    def zero_(self: Tensor) -> Tensor:
        """
        Set all element to zero in this tensor

        Returns
        -------
        Tensor
            Return this tensor itself.
        """
        self.data.zero_()
        return self

    def sqrt(self: Tensor) -> Tensor:
        """
        Get the sqrt of the tensor.

        Returns
        -------
        Tensor
            The sqrt of this tensor.
        """
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=torch.sqrt(torch.abs(self.data)),
            mask=self.mask,
        )

    def reciprocal(self: Tensor) -> Tensor:
        """
        Get the reciprocal of the tensor.

        Returns
        -------
        Tensor
            The reciprocal of this tensor.
        """
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=torch.where(self.data == 0, self.data, 1 / self.data),
            mask=self.mask,
        )

    def range(self: Tensor, first: typing.Any = 0, step: typing.Any = 1) -> Tensor:
        """
        A useful function to Get tensor filled with simple data for test in the same shape.

        Parameters
        ----------
        first, step
            Parameters to generate data.

        Returns
        -------
        Tensor
            Returns the tensor filled with simple data for test.
        """
        data = torch.cumsum(self.mask.reshape([-1]), dim=0, dtype=self.dtype).reshape(self.data.size())
        data = (data - 1) * step + first
        data = torch.where(self.mask, data, torch.zeros([], dtype=self.dtype))
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=self.mask,
        )

    def to(self: Tensor, new_type: typing.Any) -> Tensor:
        """
        Convert this tensor to other scalar type.

        Parameters
        ----------
        new_type
            The scalar data type of the new tensor.
        """
        # pylint: disable=invalid-name
        if new_type is int:
            new_type = torch.int64
        if new_type is float:
            new_type = torch.float64
        if new_type is complex:
            new_type = torch.complex128
        if isinstance(new_type, str):
            if new_type in ["float32", "S"]:
                new_type = torch.float32
            elif new_type in ["float64", "float", "D"]:
                new_type = torch.float64
            elif new_type in ["complex64", "C"]:
                new_type = torch.complex64
            elif new_type in ["complex128", "complex", "Z"]:
                new_type = torch.complex128
        if self.dtype.is_complex and new_type.is_floating_point:
            data = self.data.real.to(new_type)
        else:
            data = self.data.to(new_type)
        return Tensor(
            names=self.names,
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=self.mask,
        )

    def __init__(
        self: Tensor,
        names: list[str],
        edges: list[Edge],
        *,
        dtype: typing.Optional[torch.dtype] = None,
        fermion: typing.Optional[list[bool]] = None,
        dtypes: typing.Optional[list[torch.dtype]] = None,
        # The following argument is not public
        mask: typing.Optional[torch.Tensor] = None,
        data: typing.Optional[torch.Tensor] = None,
    ) -> None:
        """
        Create a tensor with specific shape.

        Parameters
        ----------
        names : list[str]
            The edge names of the tensor, which length is just the tensor rank.
        edges : list[Edge]
            The detail information of each edge, which length is just the tensor rank.
        dtype : torch.dtype, optional
            The dtype of the tensor, left it empty to let pytorch choose default dtype.
        fermion : list[bool], optional
            Whether each sub symmetry is fermionic, it could be left empty to derive from edges
        dtypes : list[torch.dtype], optional
            The base type of sub symmetry, it could be left empty to derive from edges
        """
        # Check the rank is correct in names and edges
        assert len(names) == len(edges)
        # Check whether there are duplicated names
        assert len(set(names)) == len(names)
        # If fermion not set, get it from edges
        if fermion is None:
            fermion = edges[0].fermion
        # If dtypes not set, get it from edges
        if dtypes is None:
            dtypes = edges[0].dtypes
        # Check if fermion is correct
        assert all(edge.fermion == fermion for edge in edges)
        # Check if dtypes is correct
        assert all(edge.dtypes == dtypes for edge in edges)

        self._fermion: list[bool] = fermion
        self._dtypes: list[torch.dtype] = dtypes
        self._names: list[str] = names
        self._edges: list[Edge] = edges

        self._data: torch.Tensor
        if data is None:
            if dtype is None:
                self._data = torch.zeros([edge.dimension for edge in self.edges])
            else:
                self._data = torch.zeros([edge.dimension for edge in self.edges], dtype=dtype)
        else:
            self._data = data
        assert self.data.size() == tuple(edge.dimension for edge in self.edges)
        assert dtype is None or self.dtype is dtype

        self._mask: torch.Tensor
        if mask is None:
            self._mask = self._generate_mask()
        else:
            self._mask = mask
        assert self.mask.size() == tuple(edge.dimension for edge in self.edges)
        assert self.mask.dtype is torch.bool

    def _generate_mask(self: Tensor) -> torch.Tensor:
        return functools.reduce(
            # Mask is valid if all sub symmetry give valid mask.
            torch.logical_and,
            (
                # The mask is valid if total symmetry is False or total symmetry is 0
                _utility.zero_symmetry(
                    # total sub symmetry is calculated by reduce
                    functools.reduce(
                        # The reduce operator depend on the dtype of this sub symmetry
                        _utility.add_symmetry,
                        (
                            # The sub symmetry of every edge will be reshape to be reduced.
                            _utility.unsqueeze(edge.symmetry[sub_symmetry_index], current_index, self.rank)
                            # The sub symmetry of every edge is reduced one by one
                            for current_index, edge in enumerate(self.edges)),
                        # Reduce from a rank-0 tensor
                        torch.zeros([], dtype=sub_symmetry_dtype),
                    ))
                # Calculate mask on every sub symmetry one by one
                for sub_symmetry_index, sub_symmetry_dtype in enumerate(self.dtypes)),
            # Reduce from all true mask
            torch.ones(self.data.size(), dtype=torch.bool),
        )

    @multimethod
    def _prepare_position(self: Tensor, position: tuple[int, ...]) -> tuple[int, ...]:
        indices: tuple[int, ...] = position
        assert len(indices) == self.rank
        assert all(0 <= index < edge.dimension for edge, index in zip(self.edges, indices))
        return indices

    @_prepare_position.register
    def _(self: Tensor, position: tuple[slice, ...]) -> tuple[int, ...]:
        index_by_name: dict[str, int] = {s.start: s.stop for s in position}
        indices: tuple[int, ...] = tuple(index_by_name[name] for name in self.names)
        assert len(indices) == self.rank
        assert all(0 <= index < edge.dimension for edge, index in zip(self.edges, indices))
        return indices

    @_prepare_position.register
    def _(self: Tensor, position: dict[str, int]) -> tuple[int, ...]:
        indices: tuple[int, ...] = tuple(position[name] for name in self.names)
        assert len(indices) == self.rank
        assert all(0 <= index < edge.dimension for edge, index in zip(self.edges, indices))
        return indices

    def __getitem__(self: Tensor, position: tuple[int, ...] | tuple[slice, ...] | dict[str, int]) -> typing.Any:
        """
        Get the element of the tensor

        Parameters
        ----------
        position : tuple[int, ...] | tuple[slice, ...] | dict[str, int]
            The position of the element, which could be either tuple of index directly or a map from edge name to the
            index in the corresponding edge.
        """
        indices: tuple[int, ...] = self._prepare_position(position)
        return self.data[indices]

    def __setitem__(self: Tensor, position: tuple[int, ...] | tuple[slice, ...] | dict[str, int],
                    value: typing.Any) -> None:
        """
        Set the element of the tensor

        Parameters
        ----------
        position : tuple[int, ...] | tuple[slice, ...] | dict[str, int]
            The position of the element, which could be either tuple of index directly or a map from edge name to the
            index in the corresponding edge.
        """
        indices: tuple[int, ...] = self._prepare_position(position)
        if self.mask[indices]:
            self.data[indices] = value
        else:
            raise IndexError("The indices specified are masked, so it is invalid to set item here.")

    def clear_symmetry(self: Tensor) -> Tensor:
        """
        Clear all symmetry of this tensor.

        Returns
        -------
        Tensor
            The result tensor with symmetry cleared.
        """
        # Mask must be generated again here
        # pylint: disable=no-else-return
        if any(self.fermion):
            return Tensor(
                names=self.names,
                edges=[
                    Edge(
                        fermion=[True],
                        dtypes=[torch.bool],
                        symmetry=[edge.parity],
                        dimension=edge.dimension,
                        arrow=edge.arrow,
                        parity=edge.parity,
                    ) for edge in self.edges
                ],
                fermion=[True],
                dtypes=[torch.bool],
                data=self.data,
            )
        else:
            return Tensor(
                names=self.names,
                edges=[
                    Edge(
                        fermion=[],
                        dtypes=[],
                        symmetry=[],
                        dimension=edge.dimension,
                        arrow=edge.arrow,
                        parity=edge.parity,
                    ) for edge in self.edges
                ],
                fermion=[],
                dtypes=[],
                data=self.data,
            )

    def randn_(self: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
        """
        Fill the tensor with random number in normal distribution.

        Parameters
        ----------
        mean, std : float
            The parameter of normal distribution.

        Returns
        -------
        Tensor
            Return this tensor itself.
        """
        self.data.normal_(mean, std)
        torch.where(self.mask, self.data, torch.zeros([], dtype=self.dtype), out=self.data)
        return self

    def rand_(self: Tensor, low: float = 0., high: float = 1.) -> Tensor:
        """
        Fill the tensor with random number in uniform distribution.

        Parameters
        ----------
        low, high : float
            The parameter of uniform distribution.

        Returns
        -------
        Tensor
            Return this tensor itself.
        """
        self.data.uniform_(low, high)
        torch.where(self.mask, self.data, torch.zeros([], dtype=self.dtype), out=self.data)
        return self

    def same_type_with(self: Tensor, other: Tensor) -> bool:
        """
        Check whether two tensor has the same type, that is to say they share the same symmetry type.
        """
        return self.fermion == other.fermion and self.dtypes == other.dtypes

    def same_shape_with(self: Tensor, other: Tensor, *, allow_transpose: bool = True) -> bool:
        """
        Check whether two tensor has the same shape, that is to say the only differences between them are transpose and
        data difference.
        """
        if not self.same_type_with(other):
            return False
        # pylint: disable=no-else-return
        if allow_transpose:
            return dict(zip(self.names, self.edges)) == dict(zip(other.names, other.edges))
        else:
            return self.names == other.names and self.edges == other.edges

    def edge_rename(self: Tensor, name_map: dict[str, str]) -> Tensor:
        """
        Rename edge name for this tensor.

        Parameters
        ----------
        name_map : dict[str, str]
            The name map to be used in renaming edge name.

        Returns
        -------
        Tensor
            The tensor with names renamed.
        """
        return Tensor(
            names=[name_map.get(name, name) for name in self.names],
            edges=self.edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=self.data,
            mask=self.mask,
        )

    def transpose(self: Tensor, names: list[str]) -> Tensor:
        """
        Transpose the tensor out-place.

        Parameters
        ----------
        names : list[str]
            The new edge order identified by edge names.

        Returns
        -------
        Tensor
            The transpose tensor.
        """
        if names == self.names:
            return self
        assert len(names) == len(self.names)
        assert set(names) == set(self.names)
        before_by_after = [self.names.index(name) for name in names]
        after_by_before = [names.index(name) for name in self.names]
        data = self.data.permute(before_by_after)
        mask = self.mask.permute(before_by_after)
        if any(self.fermion):
            # It is fermionic tensor
            parities_before_transpose = [
                _utility.unsqueeze(edge.parity, current_index, self.rank)
                for current_index, edge in enumerate(self.edges)
            ]
            # Generate parity by xor all inverse pairs
            parity = functools.reduce(
                torch.logical_xor,
                (
                    torch.logical_and(parities_before_transpose[i], parities_before_transpose[j])
                    # Loop every 0 <= i < j < rank
                    for j in range(self.rank)
                    for i in range(0, j)
                    if after_by_before[i] > after_by_before[j]),
                torch.zeros([], dtype=torch.bool))
            # parity True -> -x
            # parity False -> +x
            data = torch.where(parity.permute(before_by_after), -data, +data)
        return Tensor(
            names=names,
            edges=[self.edges[index] for index in before_by_after],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=mask,
        )

    def reverse_edge(
        self: Tensor,
        reversed_names: set[str],
        apply_parity: bool = False,
        parity_exclude_names: typing.Optional[set[str]] = None,
    ) -> Tensor:
        """
        Reverse some edge in the tensor.

        Parameters
        ----------
        reversed_names : set[str]
            The edge names of those edges which will be reversed
        apply_parity : bool, default=False
            Whether to apply parity caused by reversing edge, since reversing edge will generate half a sign.
        parity_exclude_names : set[str], optional
            The name of edges in the different behavior other than default set by apply_parity.

        Returns
        -------
        Tensor
            The tensor with edges reversed.
        """
        if not any(self.fermion):
            return self
        if parity_exclude_names is None:
            parity_exclude_names = set()
        assert all(name in self.names for name in reversed_names)
        assert all(name in reversed_names for name in parity_exclude_names)
        data = self.data
        if any(self.fermion):
            # Parity is xor of all valid reverse parity
            parity = functools.reduce(
                torch.logical_xor,
                (
                    _utility.unsqueeze(edge.parity, current_index, self.rank)
                    # Loop over all edge
                    for current_index, [name, edge] in enumerate(zip(self.names, self.edges))
                    # Check if this edge is reversed and if this edge will be applied parity
                    if (name in reversed_names) and (apply_parity ^ (name in parity_exclude_names))),
                torch.zeros([], dtype=torch.bool),
            )
            data = torch.where(parity, -data, +data)
        return Tensor(
            names=self.names,
            edges=[
                Edge(
                    fermion=edge.fermion,
                    dtypes=edge.dtypes,
                    symmetry=edge.symmetry,
                    dimension=edge.dimension,
                    arrow=not edge.arrow if self.names[current_index] in reversed_names else edge.arrow,
                    parity=edge.parity,
                ) for current_index, edge in enumerate(self.edges)
            ],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=self.mask,
        )

    @staticmethod
    def _split_edge_get_name_group(
        name: str,
        split_map: dict[str, list[tuple[str, Edge]]],
    ) -> list[str]:
        split_group: typing.Optional[list[tuple[str, Edge]]] = split_map.get(name, None)
        # pylint: disable=no-else-return
        if split_group is None:
            return [name]
        else:
            return [new_name for new_name, _ in split_group]

    @staticmethod
    def _split_edge_get_edge_group(
        name: str,
        edge: Edge,
        split_map: dict[str, list[tuple[str, Edge]]],
    ) -> list[Edge]:
        split_group: typing.Optional[list[tuple[str, Edge]]] = split_map.get(name, None)
        # pylint: disable=no-else-return
        if split_group is None:
            return [edge]
        else:
            return [new_edge for _, new_edge in split_group]

    def split_edge(
        self: Tensor,
        split_map: dict[str, list[tuple[str, Edge]]],
        apply_parity: bool = False,
        parity_exclude_names: typing.Optional[set[str]] = None,
    ) -> Tensor:
        """
        Split some edges in this tensor.

        Parameters
        ----------
        split_map : dict[str, list[tuple[str, Edge]]]
            The edge splitting plan.
        apply_parity : bool, default=False
            Whether to apply parity caused by splitting edge, since splitting edge will generate half a sign.
        parity_exclude_names : set[str], optional
            The name of edges in the different behavior other than default set by apply_parity.

        Returns
        -------
        Tensor
            The tensor with edges splitted.
        """
        if parity_exclude_names is None:
            parity_exclude_names = set()
        # Check the edge to be splitted can be merged by result edges.
        assert all(
            self.edge_by_name(name) == Edge.merge_edges(
                [new_edge for _, new_edge in split_result],
                fermion=self.fermion,
                dtypes=self.dtypes,
                arrow=self.edge_by_name(name).arrow,
            ) for name, split_result in split_map.items())
        assert all(name in split_map for name in parity_exclude_names)
        # Calculate the result components
        names: list[str] = functools.reduce(
            # Concat list
            operator.add,
            # If name in split_map, use the new names list, otherwise use name itself as a length-1 list
            (Tensor._split_edge_get_name_group(name, split_map) for name in self.names),
            # Reduce from [] to concat all list
            [],
        )
        edges: list[Edge] = functools.reduce(
            # Concat list
            operator.add,
            # If name in split_map, use the new edges list, otherwise use the edge itself as a length-1 list
            (Tensor._split_edge_get_edge_group(name, edge, split_map) for name, edge in zip(self.names, self.edges)),
            # Reduce from [] to concat all list
            [],
        )
        new_size = [edge.dimension for edge in edges]
        data = self.data.reshape(new_size)
        mask = self.mask.reshape(new_size)

        # Apply parity
        if any(self.fermion):
            # It is fermionic tensor, parity need to be applied
            new_rank = len(names)
            # Parity is xor of all valid splitting parity
            parity = functools.reduce(
                torch.logical_xor,
                (
                    # Apply the parity for this splitting group here
                    # It is need to perform a total transpose on this split group
                    # {sum 0<=i<j<r p(i) * p(j)} % 2 = {[sum p(i)]^2 - [sum p(i)]^2} & 2 = sum p(i) & 2
                    torch.bitwise_and(
                        functools.reduce(
                            torch.add,
                            (_utility.unsqueeze(new_edge.parity, names.index(new_name), new_rank)
                             for new_name, new_edge in split_result),
                            torch.zeros([], dtype=torch.int),
                        ), 2) != 0
                    # Loop over all splitting edge
                    for old_name, split_result in split_map.items()
                    # Only check if this edge splitting will be applied parity
                    if apply_parity ^ (old_name in parity_exclude_names)),
                torch.zeros([], dtype=torch.bool))

            data = torch.where(parity, -data, +data)

        return Tensor(
            names=names,
            edges=edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=mask,
        )

    def _merge_edge_get_names(self: Tensor, merge_map: dict[str, list[str]]) -> list[str]:
        reversed_names: list[str] = []
        for name in reversed(self.names):
            found_in_merge_map: typing.Optional[tuple[str, list[str]]] = next(
                ((new_name, old_names) for new_name, old_names in merge_map.items() if name in old_names), None)
            if found_in_merge_map is None:
                # This edge will not be merged
                reversed_names.append(name)
            else:
                new_name, old_names = found_in_merge_map
                # This edge will be merged
                if name == old_names[-1]:
                    # Add new edge only if it is the last edge
                    reversed_names.append(new_name)
        # Some edge is merged from no edges, it should be considered
        for new_name, old_names in merge_map.items():
            if not old_names:
                reversed_names.append(new_name)
        return list(reversed(reversed_names))

    @staticmethod
    def _merge_edge_get_name_group(name: str, merge_map: dict[str, list[str]]) -> list[str]:
        merge_group: typing.Optional[list[str]] = merge_map.get(name, None)
        # pylint: disable=no-else-return
        if merge_group is None:
            return [name]
        else:
            return merge_group

    def merge_edge(
        self: Tensor,
        merge_map: dict[str, list[str]],
        apply_parity: bool = False,
        parity_exclude_names: typing.Optional[set[str]] = None,
        *,
        merge_arrow: typing.Optional[dict[str, bool]] = None,
        names: typing.Optional[list[str]] = None,
    ) -> Tensor:
        """
        Merge some edges in this tensor.

        Parameters
        ----------
        merge_map : dict[str, list[str]]
            The edge merging plan.
        apply_parity : bool, default=False
            Whether to apply parity caused by merging edge, since merging edge will generate half a sign.
        parity_exclude_names : set[str], optional
            The name of edges in the different behavior other than default set by apply_parity.
        merge_arrow : dict[str, bool], optional
            For merging edge from zero edges, arrow cannot be identified automatically, it requires user set manually.
        names : list[str], optional
            The edge order of the result, sometimes user may want to specify it manually.

        Returns
        -------
        Tensor
            The tensor with edges merged.
        """
        # pylint: disable=too-many-locals
        if parity_exclude_names is None:
            parity_exclude_names = set()
        if merge_arrow is None:
            merge_arrow = {}
        assert all(all(old_name in self.names for old_name in old_names) for _, old_names in merge_map.items())
        assert all(name in merge_map for name in parity_exclude_names)
        # Two steps: 1. Transpose 2. Merge
        if names is None:
            names = self._merge_edge_get_names(merge_map)
        transposed_names: list[str] = functools.reduce(
            # Concat list
            operator.add,
            # If name in merge_map, use the old names list, otherwise use name itself as a length-1 list
            (Tensor._merge_edge_get_name_group(name, merge_map) for name in names),
            # Reduce from [] to concat all list
            [],
        )
        transposed_tensor = self.transpose(transposed_names)
        # Prepare a name to index map, since we need to look up it frequently later.
        transposed_name_map = {name: index for index, name in enumerate(transposed_tensor.names)}
        edges = [
            # If name is created by merging, call Edge.merge_edges to get the merged edge, otherwise get it directly
            # from transposed_tensor.
            Edge.merge_edges(
                edges=[transposed_tensor.edges[transposed_name_map[old_name]]
                       for old_name in merge_map[name]],
                fermion=self.fermion,
                dtypes=self.dtypes,
                arrow=merge_arrow.get(name, None),
                # If merging edge from zero edge, arrow need to be set manually
            ) if name in merge_map else transposed_tensor.edges[transposed_name_map[name]]
            # Loop over names
            for name in names
        ]
        transposed_data = transposed_tensor.data
        transposed_mask = transposed_tensor.mask

        # Apply parity
        if any(self.fermion):
            # It is fermionic tensor, parity need to be applied
            # Parity is xor of all valid merging parity
            parity = functools.reduce(
                torch.logical_xor,
                (
                    # Apply the parity for this merging group here
                    # It is need to perform a total transpose on this merging group
                    # {sum 0<=i<j<r p(i) * p(j)} % 2 = {[sum p(i)]^2 - [sum p(i)]^2} & 2 = sum p(i) & 2
                    torch.bitwise_and(
                        functools.reduce(
                            torch.add,
                            (_utility.unsqueeze(
                                transposed_tensor.edges[transposed_name_map[old_name]].parity,
                                transposed_name_map[old_name],
                                transposed_tensor.rank,
                            )
                             for old_name in old_names),
                            torch.zeros([], dtype=torch.int),
                        ), 2) != 0
                    # Loop over all merging edge
                    for new_name, old_names in merge_map.items()
                    # Only check if this edge merging will be applied parity
                    if apply_parity ^ (new_name in parity_exclude_names)),
                torch.zeros([], dtype=torch.bool))

            transposed_data = torch.where(parity, -transposed_data, +transposed_data)

        new_size = [edge.dimension for edge in edges]
        data = transposed_data.reshape(new_size)
        mask = transposed_mask.reshape(new_size)

        return Tensor(
            names=names,
            edges=edges,
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=mask,
        )

    def contract(
        self: Tensor,
        other: Tensor,
        contract_pairs: set[tuple[str, str]],
        fuse_names: typing.Optional[set[str]] = None,
    ) -> Tensor:
        """
        Contract two tensors.

        Parameters
        ----------
        other : Tensor
            Another tensor to be contracted.
        contract_pairs : set[tuple[str, str]]
            The pairs of edges to be contract between two tensors.
        fuse_names : set[str], optional
            The set of edges to be fuses.

        Returns
        -------
        Tensor
            The result contracted by two tensors.
        """
        # pylint: disable=too-many-locals
        # Only same type tensor can be contracted.
        assert self.same_type_with(other)

        if fuse_names is None:
            fuse_names = set()
        # Fuse name should not have any symmetry
        assert all(
            all(_utility.zero_symmetry(sub_symmetry)
                for sub_symmetry in self.edge_by_name(fuse_name).symmetry)
            for fuse_name in fuse_names)

        # Alias tensor
        tensor_1: Tensor = self
        tensor_2: Tensor = other

        # Check if contract edge and fuse edge compatible
        assert all(tensor_1.edge_by_name(name) == tensor_2.edge_by_name(name) for name in fuse_names)
        assert all(
            tensor_1.edge_by_name(name_1).conjugate() == tensor_2.edge_by_name(name_2)
            for name_1, name_2 in contract_pairs)

        # All tensor edges merged to three part: fuse edge, contract edge, free edge

        # Contract of tensor has 5 step:
        # 1. reverse arrow
        #    reverse all free edge and fuse edge to arrow False, without parity apply.
        #    reverse contract edge to two arrow: False(tensor_2) and True(tensor_1), only apply parity to one tensor.
        # 2. merge edge
        #    merge all edge in the same part to one edge, only apply parity to contract edge of one tensor
        #    free edge and fuse edge will not be applied parity.
        # 3. contract matrix
        #    call matrix multiply
        # 4. split edge
        #    split edge merged in step 2, without apply parity
        # 5. reverse arrow
        #    reverse arrow reversed in step 1, without parity apply

        # Step 1
        contract_names_1: set[str] = {name_1 for name_1, name_2 in contract_pairs}
        contract_names_2: set[str] = {name_2 for name_1, name_2 in contract_pairs}
        arrow_true_names_1: set[str] = {name for name, edge in zip(tensor_1.names, tensor_1.edges) if edge.arrow}
        arrow_true_names_2: set[str] = {name for name, edge in zip(tensor_2.names, tensor_2.edges) if edge.arrow}

        # tensor 1: contract_names & arrow_false | not contract_names & arrow_true -> contract_names ^ arrow_true
        tensor_1 = tensor_1.reverse_edge(contract_names_1 ^ arrow_true_names_1, False,
                                         contract_names_1 - arrow_true_names_1)
        tensor_2 = tensor_2.reverse_edge(arrow_true_names_2, False, set())

        # Step 2
        free_edges_1: list[tuple[str, Edge]] = [(name, edge)
                                                for name, edge in zip(tensor_1.names, tensor_1.edges)
                                                if name not in fuse_names and name not in contract_names_1]
        free_names_1: list[str] = [name for name, _ in free_edges_1]
        free_edges_2: list[tuple[str, Edge]] = [(name, edge)
                                                for name, edge in zip(tensor_2.names, tensor_2.edges)
                                                if name not in fuse_names and name not in contract_names_2]
        free_names_2: list[str] = [name for name, _ in free_edges_2]
        # Check which tensor is bigger, and use it to determine the fuse and contract edge order.
        ordered_fuse_edges: list[tuple[str, Edge]]
        ordered_fuse_names: list[str]
        ordered_contract_names_1: list[str]
        ordered_contract_names_2: list[str]
        if tensor_1.data.nelement() > tensor_2.data.nelement():
            # Tensor 1 larger, fit by tensor 1
            ordered_fuse_edges = [
                (name, edge) for name, edge in zip(tensor_1.names, tensor_1.edges) if name in fuse_names
            ]
            ordered_fuse_names = [name for name, _ in ordered_fuse_edges]

            # pylint: disable=unnecessary-comprehension
            contract_names_map = {name_1: name_2 for name_1, name_2 in contract_pairs}
            ordered_contract_names_1 = [name for name in tensor_1.names if name in contract_names_1]
            ordered_contract_names_2 = [contract_names_map[name] for name in ordered_contract_names_1]
        else:
            # Tensor 2 larger, fit by tensor 2
            ordered_fuse_edges = [
                (name, edge) for name, edge in zip(tensor_2.names, tensor_2.edges) if name in fuse_names
            ]
            ordered_fuse_names = [name for name, _ in ordered_fuse_edges]

            contract_names_map = {name_2: name_1 for name_1, name_2 in contract_pairs}
            ordered_contract_names_2 = [name for name in tensor_2.names if name in contract_names_2]
            ordered_contract_names_1 = [contract_names_map[name] for name in ordered_contract_names_2]

        put_contract_1_right: bool = next(
            (name in contract_names_1 for name in reversed(tensor_1.names) if name not in fuse_names), True)
        put_contract_2_right: bool = next(
            (name in contract_names_2 for name in reversed(tensor_2.names) if name not in fuse_names), False)

        tensor_1 = tensor_1.merge_edge(
            {
                "Free_1": free_names_1,
                "Contract_1": ordered_contract_names_1,
                "Fuse_1": ordered_fuse_names,
            },
            False,
            {"Contract_1"},
            merge_arrow={
                "Free_1": False,
                "Contract_1": True,
                "Fuse_1": False,
            },
            names=["Fuse_1", "Free_1", "Contract_1"] if put_contract_1_right else ["Fuse_1", "Contract_1", "Free_1"],
        )
        tensor_2 = tensor_2.merge_edge(
            {
                "Free_2": free_names_2,
                "Contract_2": ordered_contract_names_2,
                "Fuse_2": ordered_fuse_names,
            },
            False,
            set(),
            merge_arrow={
                "Free_2": False,
                "Contract_2": False,
                "Fuse_2": False,
            },
            names=["Fuse_2", "Free_2", "Contract_2"] if put_contract_2_right else ["Fuse_2", "Contract_2", "Free_2"],
        )
        # C[fuse, free1, free2] = A[fuse, free1 contract] B[fuse, contract free2]
        assert tensor_1.edge_by_name("Fuse_1") == tensor_2.edge_by_name("Fuse_2")
        assert tensor_1.edge_by_name("Contract_1").conjugate() == tensor_2.edge_by_name("Contract_2")

        # Step 3
        # The standard arrow is
        # (0, False, True) (0, False, False)
        # aka: (a b) (c d) (c+ b+) = (a d)
        # since: EPR pair order is (False True)
        # put_contract_1_right should be True
        # put_contract_2_right should be False
        # Every mismatch generate a sign
        # Total sign is
        # (!put_contract_1_right) ^ (put_contract_2_right) = put_contract_1_right == put_contract_2_right
        dtype = torch.result_type(tensor_1.data, tensor_2.data)
        data = torch.einsum(
            "b" + ("ic" if put_contract_1_right else "ci") + ",b" + ("jc" if put_contract_2_right else "cj") + "->bij",
            tensor_1.data.to(dtype=dtype), tensor_2.data.to(dtype=dtype))
        if put_contract_1_right == put_contract_2_right:
            data = torch.where(tensor_2.edge_by_name("Free_2").parity.reshape([1, 1, -1]), -data, +data)
        tensor = Tensor(
            names=["Fuse", "Free_1", "Free_2"],
            edges=[tensor_1.edge_by_name("Fuse_1"),
                   tensor_1.edge_by_name("Free_1"),
                   tensor_2.edge_by_name("Free_2")],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
        )

        # Step 4
        tensor = tensor.split_edge({
            "Fuse": ordered_fuse_edges,
            "Free_1": free_edges_1,
            "Free_2": free_edges_2
        }, False, set())

        # Step 5
        tensor = tensor.reverse_edge(
            (arrow_true_names_1 - contract_names_1) | (arrow_true_names_2 - contract_names_2),
            False,
            set(),
        )

        return tensor

    def _trace_group_edge(
        self: Tensor,
        trace_pairs: set[tuple[str, str]],
        fuse_names: dict[str, tuple[str, str]],
    ) -> tuple[
            list[str],
            list[str],
            list[str],
            list[str],
            list[str],
            list[str],
            list[int],
            list[int],
    ]:
        # pylint: disable=too-many-locals
        # pylint: disable=unnecessary-comprehension
        trace_map = {
            old_name_1: old_name_2 for old_name_1, old_name_2 in trace_pairs
        } | {
            old_name_2: old_name_1 for old_name_1, old_name_2 in trace_pairs
        }
        fuse_map = {
            old_name_1: (old_name_2, new_name) for new_name, [old_name_1, old_name_2] in fuse_names.items()
        } | {
            old_name_2: (old_name_1, new_name) for new_name, [old_name_1, old_name_2] in fuse_names.items()
        }
        reversed_trace_names_1: list[str] = []
        reversed_trace_names_2: list[str] = []
        reversed_fuse_names_1: list[str] = []
        reversed_fuse_names_2: list[str] = []
        reversed_free_names: list[str] = []
        reversed_fuse_names_result: list[str] = []
        reversed_free_index: list[int] = []
        reversed_fuse_index_result: list[int] = []
        added_names: set[str] = set()
        for index, name in zip(reversed(range(self.rank)), reversed(self.names)):
            if name not in added_names:
                trace_name: typing.Optional[str] = trace_map.get(name, None)
                fuse_name: typing.Optional[tuple[str, str]] = fuse_map.get(name, None)
                if trace_name is not None:
                    reversed_trace_names_2.append(name)
                    reversed_trace_names_1.append(trace_name)
                    added_names.add(trace_name)
                elif fuse_name is not None:
                    # fuse_name = another old name, new name
                    reversed_fuse_names_2.append(name)
                    reversed_fuse_names_1.append(fuse_name[0])
                    added_names.add(fuse_name[0])
                    reversed_fuse_names_result.append(fuse_name[1])
                    reversed_fuse_index_result.append(index)
                else:
                    reversed_free_names.append(name)
                    reversed_free_index.append(index)
        return (
            list(reversed(reversed_trace_names_1)),
            list(reversed(reversed_trace_names_2)),
            list(reversed(reversed_fuse_names_1)),
            list(reversed(reversed_fuse_names_2)),
            list(reversed(reversed_free_names)),
            list(reversed(reversed_fuse_names_result)),
            list(reversed(reversed_free_index)),
            list(reversed(reversed_fuse_index_result)),
        )

    def trace(
        self: Tensor,
        trace_pairs: set[tuple[str, str]],
        fuse_names: typing.Optional[dict[str, tuple[str, str]]] = None,
    ) -> Tensor:
        """
        Trace a tensor.

        Parameters
        ----------
        trace_pairs : set[tuple[str, str]]
            The pairs of edges to be traced
        fuse_names : dict[str, tuple[str, str]]
            The edges to be fused.

        Returns
        -------
        Tensor
            The traced tensor.
        """
        # pylint: disable=too-many-locals
        if fuse_names is None:
            fuse_names = {}
        # Fuse names should not have any symmetry
        assert all(
            all(_utility.zero_symmetry(sub_symmetry)
                for sub_symmetry in self.edge_by_name(old_name_1).symmetry)
            for new_name, [old_name_1, old_name_2] in fuse_names.items())
        # Fuse names should share the same edges
        assert all(
            self.edge_by_name(old_name_1) == self.edge_by_name(old_name_2)
            for new_name, [old_name_1, old_name_2] in fuse_names.items())
        # Trace edges should be compatible
        assert all(
            self.edge_by_name(old_name_1).conjugate() == self.edge_by_name(old_name_2)
            for old_name_1, old_name_2 in trace_pairs)

        # Split trace pairs and fuse names to two part before main part of trace.
        [
            trace_names_1,
            trace_names_2,
            fuse_names_1,
            fuse_names_2,
            free_names,
            fuse_names_result,
            free_index,
            fuse_index_result,
        ] = self._trace_group_edge(trace_pairs, fuse_names)

        # Make alias
        tensor = self

        # Tensor edges merged to 5 parts: fuse edge 1, fuse edge 2, trace edge 1, trace edge 2, free edge
        # Trace contains 5 step:
        # 1. reverse all arrow to False except trace edge 1 to be True, only apply parity to one of two trace edge
        # 2. merge all edge to 5 part, only apply parity to one of two trace edge
        # 3. trace trivial tensor
        # 4. split edge merged in step 2, without apply parity
        # 5. reverse arrow reversed in step 1, without apply parity

        # Step 1
        arrow_true_names = {name for name, edge in zip(tensor.names, tensor.edges) if edge.arrow}
        unordered_trace_names_1 = set(trace_names_1)
        tensor = tensor.reverse_edge(unordered_trace_names_1 ^ arrow_true_names, False,
                                     unordered_trace_names_1 - arrow_true_names)

        # Step 2
        free_edges: list[tuple[str,
                               Edge]] = [(name, tensor.edges[index]) for name, index in zip(free_names, free_index)]
        fuse_edges_result: list[tuple[str, Edge]] = [
            (name, tensor.edges[index]) for name, index in zip(fuse_names_result, fuse_index_result)
        ]
        tensor = tensor.merge_edge(
            {
                "Trace_1": trace_names_1,
                "Trace_2": trace_names_2,
                "Fuse_1": fuse_names_1,
                "Fuse_2": fuse_names_2,
                "Free": free_names,
            },
            False,
            {"Trace_1"},
            merge_arrow={
                "Trace_1": True,
                "Trace_2": False,
                "Fuse_1": False,
                "Fuse_2": False,
                "Free": False,
            },
            names=["Trace_1", "Trace_2", "Fuse_1", "Fuse_2", "Free"],
        )
        # B[fuse, free] = A[trace, trace, fuse, fuse, free]
        assert tensor.edges[2] == tensor.edges[3]
        assert tensor.edges[0].conjugate() == tensor.edges[1]

        # Step 3
        # As tested, the order of edges in this einsum is not important
        # ttffi->fi, fftti->fi, ffitt->fi, ttiff->if, ittff->if, ifftt->if
        data = torch.einsum("ttffi->fi", tensor.data)
        tensor = Tensor(
            names=["Fuse", "Free"],
            edges=[tensor.edges[2], tensor.edges[4]],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
        )

        # Step 4
        tensor = tensor.split_edge({
            "Fuse": fuse_edges_result,
            "Free": free_edges,
        }, False, set())

        # Step 5
        tensor = tensor.reverse_edge(
            # Free edge with arrow true
            {name for name in free_names if name in arrow_true_names} |
            # New edge from fused edge with arrow true
            {new_name for old_name, new_name in zip(fuse_names_1, fuse_names_result) if old_name in arrow_true_names},
            False,
            set(),
        )

        return tensor

    def conjugate(self: Tensor, trivial_metric: bool = False) -> Tensor:
        """
        Get the conjugate of this tensor.

        Parameters
        ----------
        trivial_metric : bool, default=False
            Fermionic tensor in network may result in non positive definite metric, set this trivial_metric to True to
            ensure the metric to be positive, but it breaks the associative law with tensor contract.

        Returns
        -------
        Tensor
            The conjugated tensor.
        """
        data = torch.conj(self.data)

        # Usually, only a full transpose sign is applied.
        # If trivial_metric is set True, parity in edges with arrow True is also applied.

        # Apply parity
        if any(self.fermion):
            # It is fermionic tensor, parity need to be applied

            # Parity is parity generated from a full transpose
            # {sum 0<=i<j<r p(i) * p(j)} % 2 = {[sum p(i)]^2 - [sum p(i)]^2} & 2 = sum p(i) & 2
            parity = torch.bitwise_and(
                functools.reduce(
                    torch.add,
                    (_utility.unsqueeze(edge.parity, current_index, self.rank)
                     for current_index, edge in enumerate(self.edges)),
                    torch.zeros([], dtype=torch.int),
                ), 2) != 0

            if trivial_metric:
                # Apply add-on parity to make metric positive, but it break associative law with tensor product.
                # The parity here is all parity with edge arrow=True
                parity_addon = functools.reduce(
                    torch.logical_xor,
                    (_utility.unsqueeze(edge.parity, current_index, self.rank)
                     for current_index, edge in enumerate(self.edges)
                     if edge.arrow),
                    torch.zeros([], dtype=torch.bool),
                )

                parity = torch.logical_xor(parity, parity_addon)

            data = torch.where(parity, -self.data, +self.data)

        return Tensor(
            names=self.names,
            edges=[edge.conjugate() for edge in self.edges],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data,
            mask=self.mask,
        )

    @staticmethod
    def _guess_edge(matrix: torch.Tensor, edge: Edge, arrow: bool) -> Edge:
        # Used in matrix decomposition: SVD and QR
        # It relies on decomposition of block tensor is also block tensor.
        # Otherwise it cannot guess the correct edge
        # QR
        #  Full rank case:
        #   QR has uniqueness with a diagonal unitary matrix freedom for full rank case,
        #   While diagonal unitary does not change the block condition. Since we know there is at least a decomposition
        #   result which is block matrix, we know all possible decomposition is blocked.
        #   Proof:
        #    shape of A is m * n
        #    if m >= n:
        #     A = [Q1 U1] [[R1] [0]] = [Q2 U2] [[R2] [0]]
        #     A is full rank => R1 and R2 are invertible
        #     Q1 R1 = Q2 R2 and (R1 R2 invertible) => Q2^dagger Q1 = R2 R1^-1, Q1^dagger Q2 = R1 R2^-1
        #     lemma: product of inverse of upper triangular matrix is also upper triangular.
        #     Q2^dagger Q1, Q1^dagger Q2 are upper triangular => Q2^dagger Q1 is upper triangular and lower triangular.
        #     => Q2^dagger Q1 is diagonal => Q2^dagger Q1 = R2 R1^-1 = S, where S is diagonal matrix.
        #     => Q1 = Q1 R1 R1^-1 = Q2 R2 R1^-1 = Q2 S => Q1 = Q2 S => S is diagonal unitary.
        #     At last, we have Q1 = Q2 S where S is a diagonal unitary matrix while S R1 = R2
        #    if m < n:
        #     A = Q1 [R1 N1] = Q2 [R2 N2], so we have Q1 R1 = Q2 R2
        #     This is the case for m = n, so Q1 = Q2 S, S R1 = R2.
        #     At last, Q1 N1 = Q2 S N1 = Q2 N2 implies S N1 = N2.
        #     Where S is diagonal unitary.
        #  Rank sufficient case:
        #   It is hard to get the conclusion. Program may break at this situation.
        # SVD
        #  For non-singular case
        #   SVD has uniqueness with a blocked unitary matrix freedom, which preserves the singular value subspace.
        #   So edge guessing fails iff there is the same singular value crossing different quantum number.
        #   In this case, program may break.
        #  Proof:
        #   Let m <= n, since it is symmetric on the dimension.
        #   A = U1 S1 V1 => U2 S2 V2 => A A^dagger = U1 S1^2 U1^dagger = U2 S2^2 dagger U2
        #   The eigenvalue is unique in descending order, while singular value is non-negative real number.
        #   => S1 = S2 = S, and for eigenvector, U1 = U2 Q where Q is a unitary matrix that [Q S] = 0
        #   => U1 S V1 = U2 S V2 = U2 Q S V1 = U2 S Q V2 => S Q V2 = S V1, while S is non-singular, so Q V2 = V1.
        #   At last, U1 = U2 Q, S1 = S2, Q V1 = V2.
        #  For singular case
        #   It is not determined for singular part of unitary. It is similar to the non-similar case.
        #   But at last step, S Q V2 = S V1 => Q' V2 = V1, where Q' is the same to Q only in non-singular part.
        #   So, it does break blocks only if blocks has been broken by the same singular value.
        # pylint: disable=invalid-name
        m, n = matrix.size()
        assert edge.dimension == m
        argmax = torch.argmax(matrix, dim=0)
        assert argmax.size() == (n,)
        return Edge(
            fermion=edge.fermion,
            dtypes=edge.dtypes,
            symmetry=[_utility.neg_symmetry(sub_symmetry[argmax]) for sub_symmetry in edge.symmetry],
            dimension=n,
            arrow=arrow,
            parity=edge.parity[argmax],
        )

    def _ensure_mask(self: Tensor) -> None:
        """
        Currently this function is only called from SVD decomposition. It ensure that element at mask False is very
        small and set them exactly zero.

        Any function other than SVD and QR would not break blocked tensor, while QR is implemented by givens rotation
        which preserve the blocks, so there is not need to ensure mask there.
        """
        assert torch.allclose(torch.where(self.mask, torch.zeros([], dtype=self.dtype), self.data),
                              torch.zeros([], dtype=self.dtype))
        self._data = torch.where(self.mask, self.data, torch.zeros([], dtype=self.dtype))

    def svd(
        self: Tensor,
        free_names_u: set[str],
        common_name_u: str,
        common_name_v: str,
        singular_name_u: str,
        singular_name_v: str,
        cut: int = -1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        SVD decomposition a tensor. Because of the edge created by SVD is guessed based on the SVD result, the program
        may break if there is repeated singular value which may result in non-blocked composition result.

        Parameters
        ----------
        free_names_u : set[str]
            Free names in U tensor of the result of SVD.
        common_name_u, common_name_v, singular_name_u, singular_name_v : str
            The name of generated edges.
        cut : int, default=-1
            The cut for the singular values.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            U, S, V tensor, the result of SVD.
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        free_names_v = {name for name in self.names if name not in free_names_u}

        assert all(name in self.names for name in free_names_u)
        assert common_name_u not in free_names_u
        assert common_name_v not in free_names_v

        arrow_true_names = {name for name, edge in zip(self.names, self.edges) if edge.arrow}

        tensor = self.reverse_edge(arrow_true_names, False, set())

        ordered_free_edges_u: list[tuple[str, Edge]] = [
            (name, edge) for name, edge in zip(tensor.names, tensor.edges) if name in free_names_u
        ]
        ordered_free_edges_v: list[tuple[str, Edge]] = [
            (name, edge) for name, edge in zip(tensor.names, tensor.edges) if name in free_names_v
        ]
        ordered_free_names_u: list[str] = [name for name, _ in ordered_free_edges_u]
        ordered_free_names_v: list[str] = [name for name, _ in ordered_free_edges_v]

        put_v_right = next((name in free_names_v for name in reversed(tensor.names)), True)
        tensor = tensor.merge_edge(
            {
                "SVD_U": ordered_free_names_u,
                "SVD_V": ordered_free_names_v
            },
            False,
            set(),
            merge_arrow={
                "SVD_U": False,
                "SVD_V": False
            },
            names=["SVD_U", "SVD_V"] if put_v_right else ["SVD_V", "SVD_U"],
        )

        # if self.fermion:
        #     data_1, data_s, data_2 = manual_svd(tensor.data, 1e-6)
        # else:
        #     data_1, data_s, data_2 = torch.linalg.svd(tensor.data, full_matrices=False)
        data_1, data_s, data_2 = torch.linalg.svd(tensor.data, full_matrices=False)

        if cut != -1:
            data_1 = data_1[:, :cut]
            data_s = data_s[:cut]
            data_2 = data_2[:cut, :]
        data_s = torch.diag_embed(data_s)

        free_edge_1 = tensor.edges[0]
        common_edge_1 = Tensor._guess_edge(torch.abs(data_1), free_edge_1, True)
        tensor_1 = Tensor(
            names=["SVD_U", common_name_u] if put_v_right else ["SVD_V", common_name_v],
            edges=[free_edge_1, common_edge_1],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data_1,
        )
        tensor_1._ensure_mask()  # pylint: disable=protected-access
        free_edge_2 = tensor.edges[1]
        common_edge_2 = Tensor._guess_edge(torch.abs(data_2).transpose(0, 1), free_edge_2, False)
        tensor_2 = Tensor(
            names=[common_name_v, "SVD_V"] if put_v_right else [common_name_u, "SVD_U"],
            edges=[common_edge_2, free_edge_2],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data_2,
        )
        tensor_2._ensure_mask()  # pylint: disable=protected-access
        assert common_edge_1.conjugate() == common_edge_2
        tensor_s = Tensor(
            names=[singular_name_u, singular_name_v] if put_v_right else [singular_name_v, singular_name_u],
            edges=[common_edge_2, common_edge_1],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data_s,
        )

        tensor_u = tensor_1 if put_v_right else tensor_2
        tensor_v = tensor_2 if put_v_right else tensor_1

        tensor_u = tensor_u.split_edge({"SVD_U": ordered_free_edges_u}, False, set())
        tensor_v = tensor_v.split_edge({"SVD_V": ordered_free_edges_v}, False, set())

        tensor_u = tensor_u.reverse_edge(arrow_true_names & free_names_u, False, set())
        tensor_v = tensor_v.reverse_edge(arrow_true_names & free_names_v, False, set())

        return tensor_u, tensor_s, tensor_v

    def qr(
        self: Tensor,
        free_names_direction: str,
        free_names: set[str],
        common_name_q: str,
        common_name_r: str,
    ) -> tuple[Tensor, Tensor]:
        """
        QR decomposition on this tensor. Because of the edge created by QR is guessed based on the QR result, the
        program may break if the tensor is rank deficient which may result in non-blocked composition result.

        Parameters
        ----------
        free_names_direction : 'Q' | 'q' | 'R' | 'r'
            Specify which direction the free_names will set
        free_names : set[str]
            The names of free edges after QR decomposition.
        common_name_q, common_name_r : str
            The names of edges created by QR decomposition.

        Returns
        -------
        tuple[Tensor, Tensor]
            Tensor Q and R, the result of QR decomposition.
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals

        if free_names_direction in {'Q', 'q'}:
            free_names_q = free_names
            free_names_r = {name for name in self.names if name not in free_names}
        elif free_names_direction in {'R', 'r'}:
            free_names_r = free_names
            free_names_q = {name for name in self.names if name not in free_names}

        assert all(name in self.names for name in free_names)
        assert common_name_q not in free_names_q
        assert common_name_r not in free_names_r

        arrow_true_names = {name for name, edge in zip(self.names, self.edges) if edge.arrow}

        tensor = self.reverse_edge(arrow_true_names, False, set())

        ordered_free_edges_q: list[tuple[str, Edge]] = [
            (name, edge) for name, edge in zip(tensor.names, tensor.edges) if name in free_names_q
        ]
        ordered_free_edges_r: list[tuple[str, Edge]] = [
            (name, edge) for name, edge in zip(tensor.names, tensor.edges) if name in free_names_r
        ]
        ordered_free_names_q: list[str] = [name for name, _ in ordered_free_edges_q]
        ordered_free_names_r: list[str] = [name for name, _ in ordered_free_edges_r]

        # pytorch does not provide LQ, so always put r right here
        tensor = tensor.merge_edge(
            {
                "QR_Q": ordered_free_names_q,
                "QR_R": ordered_free_names_r
            },
            False,
            set(),
            merge_arrow={
                "QR_Q": False,
                "QR_R": False
            },
            names=["QR_Q", "QR_R"],
        )

        # if self.fermion:
        #     # Blocked tensor, use Givens rotation
        #     data_q, data_r = givens_qr(tensor.data)
        # else:
        #     # Non-blocked tensor, use Householder reflection
        #     data_q, data_r = torch.linalg.qr(tensor.data, mode="reduced")
        data_q, data_r = torch.linalg.qr(tensor.data, mode="reduced")

        free_edge_q = tensor.edges[0]
        common_edge_q = Tensor._guess_edge(torch.abs(data_q), free_edge_q, True)
        tensor_q = Tensor(
            names=["QR_Q", common_name_q],
            edges=[free_edge_q, common_edge_q],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data_q,
        )
        tensor_q._ensure_mask()  # pylint: disable=protected-access
        free_edge_r = tensor.edges[1]
        # common_edge_r = Tensor._guess_edge(torch.abs(data_r).transpose(0, 1), free_edge_r, False)
        # Sometimes R matrix maybe singular, guess edge will return arbitary symmetry, so do not use guessed edge.
        common_edge_r = common_edge_q.conjugate()
        tensor_r = Tensor(
            names=[common_name_r, "QR_R"],
            edges=[common_edge_r, free_edge_r],
            fermion=self.fermion,
            dtypes=self.dtypes,
            data=data_r,
        )
        tensor_r._ensure_mask()  # pylint: disable=protected-access
        assert common_edge_q.conjugate() == common_edge_r

        tensor_q = tensor_q.split_edge({"QR_Q": ordered_free_edges_q}, False, set())
        tensor_r = tensor_r.split_edge({"QR_R": ordered_free_edges_r}, False, set())

        tensor_q = tensor_q.reverse_edge(arrow_true_names & free_names_q, False, set())
        tensor_r = tensor_r.reverse_edge(arrow_true_names & free_names_r, False, set())

        return tensor_q, tensor_r

    def identity(self: Tensor, pairs: set[tuple[str, str]]) -> Tensor:
        """
        Get the identity tensor with same shape to this tensor.

        Parameters
        ----------
        pairs : set[tuple[str, str]]
            The pair of edge names to specify the relation among edges to set identity tensor.

        Returns
        -------
        Tensor
            The result identity tensor.
        """
        # The order of edges before setting identity should be (False True)
        # Merge tensor directly to two edge, set identity and split it directly.
        # When splitting, only apply parity to one part of edges

        # pylint: disable=unnecessary-comprehension
        pairs_map = {name_1: name_2 for name_1, name_2 in pairs} | {name_2: name_1 for name_1, name_2 in pairs}
        added_names: set[str] = set()
        reversed_names_1: list[str] = []
        reversed_names_2: list[str] = []
        for name in reversed(self.names):
            if name not in added_names:
                another_name = pairs_map[name]
                reversed_names_2.append(name)
                reversed_names_1.append(another_name)
                added_names.add(another_name)
        names_1 = list(reversed(reversed_names_1))
        names_2 = list(reversed(reversed_names_2))
        # unordered_names_1 = set(names_1)
        unordered_names_2 = set(names_2)

        arrow_true_names = {name for name, edge in zip(self.names, self.edges) if edge.arrow}

        # Two edges, arrow of two edges are (False, True)
        tensor = self.reverse_edge(unordered_names_2 ^ arrow_true_names, False, unordered_names_2 - arrow_true_names)

        edges_1 = [(name, tensor.edge_by_name(name)) for name in names_1]
        edges_2 = [(name, tensor.edge_by_name(name)) for name in names_2]

        tensor = tensor.merge_edge(
            {
                "Identity_1": names_1,
                "Identity_2": names_2
            },
            False,
            {"Identity_2"},
            merge_arrow={
                "Identity_1": False,
                "Identity_2": True
            },
            names=["Identity_1", "Identity_2"],
        )

        tensor = Tensor(
            names=tensor.names,
            edges=tensor.edges,
            fermion=tensor.fermion,
            dtypes=tensor.dtypes,
            data=torch.eye(*tensor.data.size()),
            mask=tensor.mask,
        )

        tensor = tensor.split_edge({"Identity_1": edges_1, "Identity_2": edges_2}, False, {"Identity_2"})

        tensor = tensor.reverse_edge(unordered_names_2 ^ arrow_true_names, False, unordered_names_2 - arrow_true_names)

        return tensor

    def exponential(self: Tensor, pairs: set[tuple[str, str]]) -> Tensor:
        """
        Get the exponential tensor of this tensor.

        Parameters
        ----------
        pairs : set[tuple[str, str]]
            The pair of edge names to specify the relation among edges to calculate exponential tensor.

        Returns
        -------
        Tensor
            The result exponential tensor.
        """
        # The order of edges before setting exponential should be (False True)
        # Merge tensor directly to two edge, set exponential and split it directly.
        # When splitting, only apply parity to one part of edges

        unordered_names_1 = {name_1 for name_1, name_2 in pairs}
        unordered_names_2 = {name_2 for name_1, name_2 in pairs}
        if self.names and self.names[-1] in unordered_names_1:
            unordered_names_1, unordered_names_2 = unordered_names_2, unordered_names_1
        # pylint: disable=unnecessary-comprehension
        pairs_map = {name_1: name_2 for name_1, name_2 in pairs} | {name_2: name_1 for name_1, name_2 in pairs}
        reversed_names_1: list[str] = []
        reversed_names_2: list[str] = []
        for name in reversed(self.names):
            if name in unordered_names_2:
                another_name = pairs_map[name]
                reversed_names_2.append(name)
                reversed_names_1.append(another_name)
        names_1 = list(reversed(reversed_names_1))
        names_2 = list(reversed(reversed_names_2))

        arrow_true_names = {name for name, edge in zip(self.names, self.edges) if edge.arrow}

        # Two edges, arrow of two edges are (False, True)
        tensor = self.reverse_edge(unordered_names_2 ^ arrow_true_names, False, unordered_names_2 - arrow_true_names)

        edges_1 = [(name, tensor.edge_by_name(name)) for name in names_1]
        edges_2 = [(name, tensor.edge_by_name(name)) for name in names_2]

        tensor = tensor.merge_edge(
            {
                "Exponential_1": names_1,
                "Exponential_2": names_2
            },
            False,
            {"Exponential_2"},
            merge_arrow={
                "Exponential_1": False,
                "Exponential_2": True
            },
            names=["Exponential_1", "Exponential_2"],
        )

        tensor = Tensor(
            names=tensor.names,
            edges=tensor.edges,
            fermion=tensor.fermion,
            dtypes=tensor.dtypes,
            data=torch.linalg.matrix_exp(tensor.data),
            mask=tensor.mask,
        )

        tensor = tensor.split_edge({"Exponential_1": edges_1, "Exponential_2": edges_2}, False, {"Exponential_2"})

        tensor = tensor.reverse_edge(unordered_names_2 ^ arrow_true_names, False, unordered_names_2 - arrow_true_names)

        return tensor
