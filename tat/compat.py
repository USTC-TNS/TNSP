"""
This file implements a compat layer for legacy TAT interface.
"""

from __future__ import annotations
import typing
from multimethod import multimethod
import torch
from .edge import Edge as E
from .tensor import Tensor as T

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=redefined-outer-name


class Symmetry(tuple):
    """
    The compat symmetry constructor, without detailed type check.
    """

    def __new__(cls: type[Symmetry], *sym: typing.Any) -> Symmetry:
        if len(sym) == 1 and isinstance(sym[0], tuple):
            sym = sym[0]
        return tuple.__new__(Symmetry, sym)

    def __neg__(self: Symmetry) -> Symmetry:
        return Symmetry(tuple(sub_sym if isinstance(sub_sym, bool) else -sub_sym for sub_sym in self))


class CompatSymmetry:
    """
    The common Symmetry namespace.
    """

    def __init__(self: CompatSymmetry, fermion: list[bool], dtypes: list[torch.dtype]) -> None:
        # This create fake module like TAT.No, TAT.Z2 or similar things, it need to specify the symmetry attributes.
        # symmetry is set by two attributes: fermion and dtypes.
        self.fermion: list[bool] = fermion
        self.dtypes: list[torch.dtype] = dtypes

        # pylint: disable=invalid-name
        self.S: CompatScalar
        self.D: CompatScalar
        self.C: CompatScalar
        self.Z: CompatScalar
        self.float32: CompatScalar
        self.float64: CompatScalar
        self.float: CompatScalar
        self.complex64: CompatScalar
        self.complex128: CompatScalar
        self.complex: CompatScalar

        # In old TAT, something like TAT.No.D is a sub module for tensor with specific scalar type.
        # In this compat library, it is implemented by another fake module: CompatScalar.
        self.S = self.float32 = CompatScalar(self, torch.float32)
        self.D = self.float64 = self.float = CompatScalar(self, torch.float64)
        self.C = self.complex64 = CompatScalar(self, torch.complex64)
        self.Z = self.complex128 = self.complex = CompatScalar(self, torch.complex128)

        self.Edge: CompatEdge = CompatEdge(self)
        self.Symmetry: type[Symmetry] = Symmetry


class CompatEdge:
    """
    The compat edge constructor.
    """

    def __init__(self: CompatEdge, owner: CompatSymmetry) -> None:
        self.fermion: list[bool] = owner.fermion
        self.dtypes: list[torch.dtype] = owner.dtypes

    def _parse_segments(self: CompatEdge, segments: list) -> tuple[list[torch.Tensor], int]:
        # In TAT, user could use [Sym] or [(Sym, Size)] to set segments of a edge, where [(Sym, Size)] is nothing but
        # the symmetry and size of every blocks. While [Sym] acts like [(Sym, 1)], so try to treat input as
        # [(Sym, Size)] First, if error raised, convert it from [Sym] to [(Sym, 1)] and try again.
        try:
            # try [(Sym, Size)] first
            return self._parse_segments_kernel(segments)
        except TypeError:
            # Cannot unpack is a type error, value[index] is a type error, too. So only catch TypeError here.
            # convert [Sym] to [(Sym, Size)]
            return self._parse_segments_kernel([(sym, 1) for sym in segments])
        # This function return the symmetry list and dimension

    def _parse_segments_kernel(
        self: CompatEdge,
        segments: list[tuple[typing.Any, int]],
    ) -> tuple[list[torch.Tensor], int]:
        # [(Sym, Size)] for every element
        dimension = sum(dim for _, dim in segments)
        symmetry = [
            torch.tensor(
                # tat.Edge need torch.Tensor as its symmetry, convert it to torch.Tensor with specific dtype.
                sum(
                    # Concat all segment for this sub symmetry from an empty list
                    # Every segment is just sym[index] * dim, sometimes sym may be sub symmetry itself directly instead
                    # of tuple of sub symmetry, so call an utility function _parse_segments_get_subsymmetry here.
                    ([self._parse_segments_get_subsymmetry(sym, index)] * dim
                     for sym, dim in segments),
                    [],
                ),
                dtype=sub_symmetry,
            )
            # Generate sub symmetry one by one
            for index, sub_symmetry in enumerate(self.dtypes)
        ]
        return symmetry, dimension

    def _parse_segments_get_subsymmetry(self: CompatEdge, sym: object, index: int) -> object:
        # Most of time, symmetry is a tuple of sub symmetry
        # But if there is only one sub symmetry in the symmetry, it could not be a tuple but sub symmetry itself.
        # pylint: disable=no-else-return
        if isinstance(sym, tuple):
            # If it is tuple, there is no need to do any other check
            return sym[index]
        else:
            # If it is not tuple, it should be sub symmetry directly, so this symmetry only should own single sub
            # symmetry, check it.
            if len(self.fermion) == 1:
                return sym
            else:
                raise TypeError(f"{sym=} is not subscript-able")

    @multimethod
    def __call__(self: CompatEdge, edge: E) -> E:
        """
        Create edge with compat interface.

        It may be created by
        1. Edge(dimension) create trivial symmetry with specified dimension.
        2. Edge(segments, arrow) create with given segments and arrow.
        3. Edge(segments_arrow_tuple) create with a tuple of segments and arrow.
        """
        # pylint: disable=invalid-name
        return edge

    @__call__.register
    def _(self: CompatEdge, dimension: int) -> E:
        # Generate a trivial symmetry tuple. In this tuple, every sub symmetry is a torch.zeros tensor with specific
        # dtype and the same dimension.
        symmetry = [torch.zeros(dimension, dtype=sub_symmetry) for sub_symmetry in self.dtypes]
        return E(fermion=self.fermion, dtypes=self.dtypes, symmetry=symmetry, dimension=dimension, arrow=False)

    @__call__.register
    def _(self: CompatEdge, segments: list, arrow: bool = False) -> E:
        symmetry, dimension = self._parse_segments(segments)
        return E(fermion=self.fermion, dtypes=self.dtypes, symmetry=symmetry, dimension=dimension, arrow=arrow)

    @__call__.register
    def _(self: CompatEdge, segments_and_bool: tuple[list, bool]) -> E:
        segments, arrow = segments_and_bool
        symmetry, dimension = self._parse_segments(segments)
        return E(fermion=self.fermion, dtypes=self.dtypes, symmetry=symmetry, dimension=dimension, arrow=arrow)


class CompatScalar:
    """
    The common Scalar namespace.
    """

    def __init__(self: CompatScalar, symmetry: CompatSymmetry, dtype: torch.dtype) -> None:
        # This is fake module like TAT.No.D, TAT.Fermi.complex, so it records the parent symmetry information and its
        # own dtype.
        self.symmetry: CompatSymmetry = symmetry
        self.dtype: torch.dtype = dtype
        # pylint: disable=invalid-name
        self.Tensor: CompatTensor = CompatTensor(self)


class CompatTensor:
    """
    The compat tensor constructor.
    """

    def __init__(self: CompatTensor, owner: CompatScalar) -> None:
        self.symmetry: CompatSymmetry = owner.symmetry
        self.dtype: torch.dtype = owner.dtype
        self.model: CompatSymmetry = owner.symmetry
        self.is_complex: bool = self.dtype.is_complex
        self.is_real: bool = self.dtype.is_floating_point

    @multimethod
    def __call__(self: CompatTensor, tensor: T) -> T:
        """
        Create tensor with compat names and edges.

        It may be create by
        1. Tensor(names, edges) The most used interface.
        2. Tensor() Create a rank-0 tensor, fill with number 1.
        3. Tensor(number, names=[], edge_symmetry=[], edge_arrow=[]) Create a size-1 tensor, with specified edge, and
           filled with specified number.
        """
        # pylint: disable=invalid-name
        return tensor

    @__call__.register
    def _(self: CompatTensor, names: list[str], edges: list) -> T:
        return T(
            names,
            [self.symmetry.Edge(edge) for edge in edges],
            fermion=self.symmetry.fermion,
            dtypes=self.symmetry.dtypes,
            dtype=self.dtype,
        )

    @__call__.register
    def _(self: CompatTensor) -> T:
        result = T(
            [],
            [],
            fermion=self.symmetry.fermion,
            dtypes=self.symmetry.dtypes,
            data=torch.ones([], dtype=self.dtype),
        )
        return result

    @__call__.register
    def _(
        self: CompatTensor,
        number: typing.Any,
        names: typing.Optional[list[str]] = None,
        edge_symmetry: typing.Optional[list] = None,
        edge_arrow: typing.Optional[list[bool]] = None,
    ) -> T:
        # Create high rank tensor with only one element
        if names is None:
            names = []
        if edge_symmetry is None:
            edge_symmetry = [None for _ in names]
        if edge_arrow is None:
            edge_arrow = [False for _ in names]
        result = T(
            names,
            [
                # Create edge for every rank, given the only symmetry(maybe None) and arrow.
                E(
                    fermion=self.symmetry.fermion,
                    dtypes=self.symmetry.dtypes,
                    # For every edge, its symmetry is a list of all sub symmetry.
                    symmetry=[
                        # For every sub symmetry, get the only symmetry for it, since dimension of all edge is 1.
                        # It should be noticed that the symmetry may be None, tuple or sub symmetry directly.
                        torch.tensor([self._create_size1_get_subsymmetry(symmetry, index)], dtype=dtype)
                        for index, dtype in enumerate(self.symmetry.dtypes)
                    ],
                    dimension=1,
                    arrow=arrow,
                )
                for symmetry, arrow in zip(edge_symmetry, edge_arrow)
            ],
            fermion=self.symmetry.fermion,
            dtypes=self.symmetry.dtypes,
            data=torch.full([1 for _ in names], number, dtype=self.dtype),
        )
        return result

    def _create_size1_get_subsymmetry(self: CompatTensor, sym: object, index: int) -> object:
        # pylint: disable=no-else-return
        # sym may be None, tuple or sub symmetry directly.
        if sym is None:
            # If is None, user may want to create symmetric edge with trivial symmetry, which should be 0 for int and
            # False for bool, always return 0 here, since it will be converted to correct type by torch.tensor.
            return 0
        elif isinstance(sym, tuple):
            # If it is tuple, there is no need to do any other check
            return sym[index]
        else:
            # If it is not tuple, it should be sub symmetry directly, so this symmetry only should own single sub
            # symmetry, check it.
            if len(self.symmetry.fermion) == 1:
                return sym
            else:
                raise TypeError(f"{sym=} is not subscript-able")


# Create fake sub module for all symmetry compiled in old version TAT
No: CompatSymmetry = CompatSymmetry(fermion=[], dtypes=[])
Z2: CompatSymmetry = CompatSymmetry(fermion=[False], dtypes=[torch.bool])
U1: CompatSymmetry = CompatSymmetry(fermion=[False], dtypes=[torch.int])
Fermi: CompatSymmetry = CompatSymmetry(fermion=[True], dtypes=[torch.int])
FermiZ2: CompatSymmetry = CompatSymmetry(fermion=[True, False], dtypes=[torch.int, torch.bool])
FermiU1: CompatSymmetry = CompatSymmetry(fermion=[True, False], dtypes=[torch.int, torch.int])
Parity: CompatSymmetry = CompatSymmetry(fermion=[True], dtypes=[torch.bool])
FermiFermi: CompatSymmetry = CompatSymmetry(fermion=[True, True], dtypes=[torch.int, torch.int])
Normal: CompatSymmetry = No

# SJ Dong's convention


def arrow(int_arrow: int) -> bool:
    "SJ Dong's convention of arrow"
    # pylint: disable=no-else-return
    if int_arrow == +1:
        return False
    elif int_arrow == -1:
        return True
    else:
        raise ValueError("int arrow should be +1 or -1.")


def parity(int_parity: int) -> bool:
    "SJ Dong's convention of parity"
    # pylint: disable=no-else-return
    if int_parity == +1:
        return False
    elif int_parity == -1:
        return True
    else:
        raise ValueError("int parity should be +1 or -1.")


# Segment index


@T._prepare_position.register  # pylint: disable=protected-access,no-member
def _(self: T, position: dict[str, tuple[typing.Any, int]]) -> tuple[int, ...]:
    return tuple(index_by_point(edge, position[name]) for name, edge in zip(self.names, self.edges))


# Add some compat interface


def _compat_function(
    focus_type: type,
    name: typing.Optional[str] = None,
) -> typing.Callable[[typing.Callable], typing.Callable]:

    def _result(function: typing.Callable) -> typing.Callable:
        if name is None:
            attr_name = function.__name__
        else:
            attr_name = name
        setattr(focus_type, attr_name, function)
        return function

    return _result


@property  # type: ignore[misc]
def storage(self: T) -> typing.Any:
    "Get the storage of the tensor"
    assert self.data.is_contiguous()
    return self.data.reshape([-1])


@_compat_function(T, name="storage")  # type: ignore[misc]
@storage.setter
def storage(self: T, value: typing.Any) -> None:
    "Set the storage of the tensor"
    assert self.data.is_contiguous()
    self.data.reshape([-1])[:] = torch.as_tensor(value)


@_compat_function(T)
def range_(self: T, first: float = 0, step: float = 1) -> T:
    "Compat Interface: Set range inplace for this tensor."
    result = self.range(first, step)
    self._data = result._data  # pylint: disable=protected-access
    return self


@_compat_function(T)
def identity_(self: T, pairs: set[tuple[str, str]]) -> T:
    "Compat Interface: Set idenity inplace for this tensor."
    result = self.identity(pairs).transpose(self.names)
    self._data = result._data  # pylint: disable=protected-access
    return self


# Exponential arguments

origin_exponential = T.exponential


@_compat_function(T)
def exponential(self: T, pairs: set[tuple[str, str]], step: typing.Optional[int] = None) -> T:
    "Compat Interface: Get the exponential tensor of this tensor."
    # pylint: disable=unused-argument
    return origin_exponential(self, pairs)


# Edge point conversion


@_compat_function(E)
def index_by_point(self: E, point: tuple[typing.Any, int]) -> int:
    "Get index by point on an edge"
    sym, sub_index = point
    if not isinstance(sym, tuple):
        sym = (sym,)
    for total_index in range(self.dimension):
        if all(sub_sym == sub_symmetry[total_index] for sub_sym, sub_symmetry in zip(sym, self.symmetry)):
            if sub_index == 0:
                return total_index
            sub_index = sub_index - 1
    raise ValueError("Invalid input point")


@_compat_function(E)
def point_by_index(self: E, index: int) -> tuple[typing.Any, int]:
    "Get point by index on an edge"
    sym = Symmetry(tuple(sub_symmetry[index] for sub_symmetry in self.symmetry))
    sub_index = sum(
        1 for i in range(index) if all(sub_sym == sub_symmetry[i] for sub_sym, sub_symmetry in zip(sym, self.symmetry)))
    return sym, sub_index


# Random utility


class CompatRandom:
    """
    Fake module for compat random utility in TAT.
    """

    def uniform_int(self: CompatRandom, low: int, high: int) -> typing.Callable[[], int]:
        "Generator for integer uniform distribution"
        # Mypy cannot recognize item of int64 tensor is int, so cast it manually.
        return staticmethod(  # type: ignore[return-value] #  python3.9 does not treat staticmethod as callable
            lambda: int(torch.randint(low, high + 1, [], dtype=torch.int64).item()))

    def uniform_real(self: CompatRandom, low: float, high: float) -> typing.Callable[[], float]:
        "Generator for float uniform distribution"
        return staticmethod(  # type: ignore[return-value] #  python3.9 does not treat staticmethod as callable
            lambda: torch.rand([], dtype=torch.float64).item() * (high - low) + low)

    def normal(self: CompatRandom, mean: float, stddev: float) -> typing.Callable[[], float]:
        "Generator for float normal distribution"
        return staticmethod(  # type: ignore[return-value] #  python3.9 does not treat staticmethod as callable
            lambda: torch.normal(mean, stddev, [], dtype=torch.float64).item())

    def seed(self: CompatRandom, new_seed: int) -> None:
        "Set the seed for random generator manually"
        torch.manual_seed(new_seed)


random = CompatRandom()
