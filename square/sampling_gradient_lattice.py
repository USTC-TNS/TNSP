#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from typing import Any, Callable, Dict, List, Tuple
from multimethod import multimethod
import TAT
from .abstract_network_lattice import AbstractNetworkLattice
from .auxiliaries import SquareAuxiliariesSystem
from . import simple_update_lattice

__all__ = ["SamplingGradientLattice"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo


class SpinConfiguration(SquareAuxiliariesSystem):
    __slots__ = ["lattice", "configuration"]

    @multimethod
    def __init__(self, lattice: SamplingGradientLattice) -> None:
        super().__init__(lattice.M, lattice.N, lattice.dimension_cut)
        self.lattice: SamplingGradientLattice = lattice
        self.configuration: List[List[int]] = [[-1 for _ in range(self.lattice.N)] for _ in range(self.lattice.M)]

    @multimethod
    def __init__(self, other: SpinConfiguration) -> None:
        super().__init__(other)
        self.lattice: SamplingGradientLattice = other.lattice
        self.configuration: List[List[int]] = [[other.configuration[i][j] for j in range(self.lattice.N)] for i in range(self.lattice.M)]

    def __iadd__(self, other: SpinConfiguration) -> SpinConfiguration:
        if self.lattice is not other.lattice:
            raise ValueError("Different basic lattice when combining two spin configuration")
        for i in range(self._M):
            for j in range(self._N):
                if self.configuration[i][j] != -1 and other.configuration[i][j] != -1:
                    if self.configuration[i][j] != other.configuration[i][j]:
                        raise ValueError("Overlap configuration when combining two spin configuration")
        super().__iadd__(other)
        return self

    def __add__(self, other: SpinConfiguration) -> SpinConfiguration:
        result = SpinConfiguration(other)
        result += other
        return result

    def __delitem__(self, position: Tuple[int, int]) -> None:
        x, y = position
        super().__delitem__((x, y))
        self.configuration[x][y] = -1

    def __setitem__(self, position: Tuple[int, int], value: int) -> None:
        x, y = position
        if self.configuration[x][y] != value:
            super().__setitem__((x, y), self.lattice[x, y].shrink({"P": value}))
            self.configuration[x][y] = value

    for signature, function in SquareAuxiliariesSystem.__getitem__.items():
        __getitem__ = multimethod(function)

    @multimethod
    def __getitem__(self, replacement: Dict[Tuple[int, int], int]) -> Tensor:
        # TODO 也许这里还可以再来层cache
        real_replacement: Dict[Tuple[int, int], Tensor] = {}
        for [x, y], spin in replacement.items():
            if self.configuration[x][y] != spin:
                # TODO shrink的cache
                real_replacement[x, y] = self.lattice[x, y].shrink({"P": spin})
        if real_replacement:
            return super().__getitem__(real_replacement)
        else:
            return super().__getitem__(None)


class SamplingGradientLattice(AbstractNetworkLattice):
    __slots__ = ["dimension_cut", "spin"]

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, Dc: int, d: int) -> None:
        super().__init__(M, N, D=D, d=d)

        self.dimension_cut: int = Dc
        self.spin: SpinConfiguration = SpinConfiguration(self)

    @multimethod
    def __init__(self, other: SamplingGradientLattice) -> None:
        super().__init__(other)

        self.dimension_cut: int = other.dimension_cut
        self.spin: SpinConfiguration = SpinConfiguration(other.spin)

    @multimethod
    def __init__(self, other: simple_update_lattice.SimpleUpdateLattice, *, Dc: int = 2) -> None:
        super().__init__(other)
        for i in range(self.M):
            for j in range(self.N):
                to_multiple = self[i, j]
                to_multiple = other.try_multiple(to_multiple, i, j, "D")
                to_multiple = other.try_multiple(to_multiple, i, j, "R")
                self[i, j] = to_multiple

        self.dimension_cut: int = Dc
        self.spin: SpinConfiguration = SpinConfiguration(self)

    def _initialize_spin(self, function: Callable[[int, int], int]) -> None:
        for i in range(self.M):
            for j in range(self.N):
                self.spin[i, j] = function(i, j)

    @multimethod
    def initialize_spin(self, function: Callable[[int, int], int]) -> None:
        self._initialize_spin(function)

    @multimethod
    def initialize_spin(self, array: List[List[int]]) -> None:
        self._initialize_spin(lambda i, j: array[i][j])

    @multimethod
    def initialize_spin(self) -> None:
        self._initialize_spin(lambda i, j: (i + j) % self.dimension_physics)

    def _hopping_spin_single_step(self) -> None:
        ws = float(self.spin[None])
        for positions, hamiltonian in self.hamiltonian.items():
            # positions: Tuple[Tuple[int, int], ...]
            # hamiltonian: Tensor
            body: int = len(positions)
            current_spins: Tuple[int, ...] = tuple(self.spin.configuration[positions[i][0]][positions[i][1]] for i in range(body))
            possible_hopping: List[Tuple[int, ...], float] = []
            for [spins_in, spins_out], element in self._find_element(hamiltonian).items():
                if spins_in == current_spins:
                    possible_hopping.append((spins_out, element))
            if possible_hopping:
                hopping_number = len(possible_hopping)
                spins_new, element = possible_hopping[TAT.random.uniform_int(0, hopping_number - 1)]
                replacement = {positions[i]: spins_new[i] for i in range(body)}
                wss = float(self.spin[replacement])
                p = (wss**2) / (ws**2)
                if TAT.random.uniform_real(0, 1) > p:
                    ws = wss
                    for i in range(body):
                        self.spin[positions[i][0], positions[i][1]] = spins_new[i]

    # <A> = sum(s,s') w(s) A(s,s') w(s') / sum(s) w(s)^2 = < sum(s') A(s, s') w(s')/w(s) >_(w(s)^2)
    # <A> = < A_s >_(w(s)^2)
    # gradient = 2 <E_s Delta_s> - 2 <E_s> <Delta_s>
    # where Delta_s = [w(s) with hole] / w(s)
    def markov_chain(self, step: int, observers: Dict[Any, Dict[Tuple[Tuple[int, int], ...], Tensor]], *, calculate_gradient: bool = False) -> Dict[Any, Dict[Tuple[Tuple[int, int]]], Tensor]:
        # 准备结果的容器
        result: Dict[Any, Dict[Tuple[Tuple[int, int], ...], Tensor]] = {kind: {positions: 0 for positions in group} for kind, group in observers.items()}
        if calculate_gradient:
            result["gradient"] = {}
        # markov sampling
        for t in range(step):
            print("markov sampling, step =", t, end="")
            self._hopping_spin_single_step()
            ws = float(self.spin[None])
            for kind, group in observers.items():
                for positions, tensor in group.items():
                    body: int = len(positions)
                    current_spins: Tuple[int, ...] = tuple(self.spin.configuration[positions[i][0]][positions[i][1]] for i in range(body))
                    value = 0
                    for [spins_in, spins_out], element in self._find_element(tensor).items():
                        if spins_in == current_spins:
                            wss = float(self.spin[{positions[i]: spins_out[i] for i in range(body)}])
                            value += element * wss / ws
                    result[kind][positions] += value
            if calculate_gradient:
                # TODO grad
                # does energy need special treat?
                raise NotImplementedError("Gradient not implemented")
            if "Energy" in result:
                print(", Energy =", sum(result["Energy"].values()) / ((t + 1) * self.M * self.N), end="")
            print()
        return result

    tensor_element_dict: Dict[int, Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]] = {}

    # TODO complex version
    def _find_element(self, tensor: Tensor) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
        tensor_id = id(tensor)
        if tensor_id in self.tensor_element_dict:
            return self.tensor_element_dict[tensor_id]
        body = len(tensor.name) // 2
        self._check_hamiltonian_name(tensor, body)
        self.tensor_element_dict[tensor_id] = {}
        result: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = self.tensor_element_dict[tensor_id]
        names = [f"I{i}" for i in range(body)] + [f"O{i}" for i in range(body)]
        index = [0 for _ in range(2 * body)]
        while True:
            value: float = tensor[{names[i]: index[i] for i in range(2 * body)}]
            if value != 0:
                result[tuple(index[:body]), tuple(index[body:])] = value
            active_position = 0
            index[active_position] += 1
            while index[active_position] == self.dimension_physics:
                index[active_position] = 0
                active_position += 1
                if active_position == 2 * body:
                    return result
                index[active_position] += 1
