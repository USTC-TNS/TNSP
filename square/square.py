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

import pickle
from enum import Enum
from typing import Union
import numpy as np
import TAT

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo

Sx: Tensor = Tensor(["I0", "O0"], [2, 2])
Sx.block[{}] = [[0, 0.5], [0.5, 0]]
Sy: CTensor = CTensor(["I0", "O0"], [2, 2])
Sy.block[{}] = [[0, -0.5j], [0.5j, 0]]
Sz: Tensor = Tensor(["I0", "O0"], [2, 2])
Sz.block[{}] = [[0.5, 0], [0, -0.5]]

SxSx: Tensor = Sx.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sx).to(float)
SySy: Tensor = Sy.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sy).to(float)
SzSz: Tensor = Sz.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sz).to(float)

SS: Tensor = SxSx + SySy + SzSz


class StateType(Enum):
    NotSet = 0
    Exact = 1
    WithEnvironment = 2
    WithoutEnvironment = 3


class SquareLattice:
    __slots__ = ["M", "N", "dimension_physics", "dimension_virtual", "_state_type", "vector", "lattice", "environment", "hamiltonian", "spin", "_auxiliaries"]

    def __init__(self, M: int, N: int, D: int = 2, d: int = 2) -> None:
        # 系统自身信息
        self.M: int = M
        self.N: int = N
        self.dimension_physics: int = d

        # 网络信息
        self.dimension_virtual: int = D

        # 系统表示方式
        self._state_type: StateType = StateType.NotSet

        # 系统表示
        self.vector: Tensor = Tensor()
        self.lattice: list[list[Tensor]] = []
        self.environment: dict[tuple[str, int, int], Tensor] = {}  # 第一个str可以是"D"或者"R"

        # 哈密顿量
        self.hamiltonian: dict[tuple[tuple[int, int], ...], Tensor] = {}  # 可以是任意体哈密顿量, 哈密顿量的name方案参考自旋, 使用I0, I1, ..., O0, O1, ...

        # 用于网络采样的spin列表
        self.spin: list[list[int]] = []

        # 辅助张量
        self._auxiliaries: dict[tuple[str, int, int]] = {}

    def simple_update(self, time: int, delta_t: float):
        if self._state_type != StateType.WithEnvironment:
            raise ValueError("State type is not WithEnv")
        updater: dict[tuple[tuple[int, int], ...], Tensor] = {}
        for positions, term in self.hamiltonian.items():
            site_number: int = len(positions)
            updater[positions] = term.exponential({(f"I{i}", f"O{i}") for i in range(site_number)}, 8)
        for _ in range(time):
            for positions, term in updater.items():
                self._single_term_simple_update(positions, term)
            for positions, term in reversed(updater.items()):
                self._single_term_simple_update(positions, term)

    def _single_term_simple_update(self, positions: tuple[tuple[int, int], ...], updater: Tensor):
        if len(positions) == 1:
            return self._single_term_simple_update_single_site(positions[0], updater)
        if len(positions) == 2:
            position_1, position_2 = positions
            if position_1[0] == position_2[0]:
                if position_1[1] == position_2[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_2, updater)
                if position_2[1] == position_1[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_1, updater)
            if position_1[1] == position_2[1]:
                if position_1[0] == position_2[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_2, updater)
                if position_2[0] == position_1[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_1, updater)
        raise NotImplementedError("Unsupported simple update style")

    def _single_term_simple_update_double_site_nearest_horizontal(self, position: tuple[int, int], updater: Tensor):
        raise NotImplementedError()
        pass

    def _single_term_simple_update_double_site_nearest_vertical(self, position: tuple[int, int], updater: Tensor):
        raise NotImplementedError()
        pass

    def _single_term_simple_update_single_site(self, position: tuple[int, int], updater: Tensor):
        i, j = position
        self.lattice[i][j] = self.lattice[i][j].contract(updater, {("P", "I0")}).edge_rename({"O0": "P"})

    def exact_update(self, time: int, approximate_energy: float = -0.5, print_energy: bool = False) -> float:
        if self._state_type != StateType.Exact:
            raise ValueError("State type is not Exact")
        total_approximate_energy: float = abs(approximate_energy) * self.M * self.N
        energy: float = 0
        for _ in range(time):
            norm_max: float = float(self.vector.norm_max())
            energy = total_approximate_energy - norm_max
            self.vector /= norm_max
            temporary_vector: Tensor = self.vector.same_shape().zero()
            for positions, value in self.hamiltonian.items():
                this_term: Tensor = self.vector.contract_all_edge(value.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})).edge_rename(
                    {f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})
                temporary_vector += this_term
            self.vector *= total_approximate_energy
            self.vector -= temporary_vector
            # v <- a v - H v = (a - H) v => E = a - v'/v
            if print_energy:
                print(energy / (self.M * self.N))
        return energy / (self.M * self.N)

    def exact_observe(self, positions: tuple[tuple[int, int], ...], observer: Union[Tensor, CTensor]) -> float:
        if self._state_type != StateType.Exact:
            raise ValueError("State type is not Exact")
        vv: Tensor = self.vector.contract_all_edge(self.vector)
        if isinstance(observer, CTensor):
            complex_vector: CTensor = self.vector.to(complex)
            vOv: Tensor = complex_vector.contract_all_edge(observer.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})).edge_rename(
                {f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)}).contract_all_edge(complex_vector)
            return complex(vOv).real / float(vv)
        else:
            vOv: Tensor = self.vector.contract_all_edge(observer.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})).edge_rename(
                {f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)}).contract_all_edge(self.vector)
            return float(vOv) / float(vv)

    @staticmethod
    def _check_hamiltonian_name(tensor: Tensor, body: int):
        if {f"{i}" for i in tensor.name} != {f"{i}{j}" for i in ["I", "O"] for j in range(body)}:
            raise ValueError("Wrong hamiltonian name")

    @property
    def single_site_hamiltonian(self):
        raise RuntimeError("Getting hamiltonian is not allowed")

    @single_site_hamiltonian.setter
    def single_site_hamiltonian(self, value: Tensor):
        self._check_hamiltonian_name(value, 1)
        for i in range(self.M):
            for j in range(self.N):
                if ((i, j),) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[((i, j),)] = value

    @property
    def vertical_bond_hamiltonian(self):
        raise RuntimeError("Getting hamiltonian is not allowed")

    @vertical_bond_hamiltonian.setter
    def vertical_bond_hamiltonian(self, value: Tensor):
        self._check_hamiltonian_name(value, 2)
        for i in range(self.M - 1):
            for j in range(self.N):
                if ((i, j), (i + 1, j)) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[(i, j), (i + 1, j)] = value

    @property
    def horizontal_bond_hamiltonian(self):
        raise RuntimeError("Getting hamiltonian is not allowed")

    @horizontal_bond_hamiltonian.setter
    def horizontal_bond_hamiltonian(self, value: Tensor):
        self._check_hamiltonian_name(value, 2)
        for i in range(self.M):
            for j in range(self.N - 1):
                if ((i, j), (i, j + 1)) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[(i, j), (i, j + 1)] = value

    @property
    def state_type(self):
        return self._state_type

    @state_type.setter
    def state_type(self, new_state_type: StateType):
        # NotSet
        if new_state_type == StateType.NotSet:
            self.vector = Tensor()
            self.lattice = []
            self.environment = {}
        # Exact
        elif new_state_type == StateType.Exact:
            if self._state_type == StateType.NotSet:
                self._initialize_vector()
            elif self._state_type == StateType.WithEnvironment:
                self._absorb_environment()
                self.environment = {}
                self._contract_all_network()
                self.lattice = []
            elif self._state_type == StateType.WithoutEnvironment:
                self._contract_all_network()
                self.lattice = []
        # WithEnv
        elif new_state_type == StateType.WithEnvironment:
            if self._state_type == StateType.NotSet:
                self._initialize_network()
                self._construct_environment()
            elif self._state_type == StateType.Exact:
                raise ValueError("Cannot Convert WithEnv to Exact")
            elif self._state_type == StateType.WithoutEnvironment:
                self._construct_environment()
        # WithoutEnv
        else:
            if self._state_type == StateType.NotSet:
                self._initialize_network()
            elif self._state_type == StateType.Exact:
                raise ValueError("Cannot Convert WithoutEnv to Exact")
            elif self._state_type == StateType.WithEnvironment:
                self._absorb_environment()
                self.environment = {}

        # Update value
        self._state_type = new_state_type

    def _initialize_vector(self):
        name_list = [f"P-{i}-{j}" for i in range(self.M) for j in range(self.N)]
        dimension_list = [self.dimension_physics for _ in range(self.M) for _ in range(self.N)]
        self.vector = Tensor(name_list, dimension_list).randn()

    def _initialize_network(self):
        self.lattice = [[self._initialize_tensor_in_network(i, j) for j in range(self.N)] for i in range(self.M)]

    def _initialize_tensor_in_network(self, i: int, j: int):
        name_list = ["P"]
        dimension_list = [self.dimension_physics]
        if i != 0:
            name_list.append("U")
            dimension_list.append(self.dimension_virtual)
        if j != 0:
            name_list.append("L")
            dimension_list.append(self.dimension_virtual)
        if i != self.M - 1:
            name_list.append("D")
            dimension_list.append(self.dimension_virtual)
        if j != self.N - 1:
            name_list.append("R")
            dimension_list.append(self.dimension_virtual)
        return Tensor(name_list, dimension_list).randn()

    def _absorb_environment(self):
        for i in range(self.M):
            for j in range(self.N - 1):
                self.lattice[i][j] = self.lattice[i][j].multiple(self.environment[("R", i, j)], "R", "U")

        for i in range(self.M - 1):
            for j in range(self.N):
                self.lattice[i][j] = self.lattice[i][j].multiple(self.environment[("D", i, j)], "D", "U")

    def _construct_environment(self):
        for i in range(self.M):
            for j in range(self.N - 1):
                self.environment[("R", i, j)] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

        for i in range(self.M - 1):
            for j in range(self.N):
                self.environment[("D", i, j)] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

    def _contract_all_network(self):
        self.vector = Tensor(1)
        for i in range(self.M):
            for j in range(self.N):
                self.vector = self.vector.contract(self.lattice[i][j].edge_rename({"D": f"D-{j}", "P": f"P-{i}-{j}"}), {("R", "L"), (f"D-{j}", "U")})


lattice = SquareLattice(3, 3)
lattice.state_type = StateType.WithEnvironment
lattice.horizontal_bond_hamiltonian = SS
lattice.vertical_bond_hamiltonian = SS
lattice.simple_update(10, 0.1)
