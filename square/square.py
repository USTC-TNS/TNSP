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
from typing import List, Optional, Tuple, Dict
import numpy as np
import TAT

CTensor = TAT(complex)
Tensor = TAT(float)

Sx = CTensor(["I", "O"], [2, 2])
Sx.block[{}] = [[0, 0.5], [0.5, 0]]
Sy = CTensor(["I", "O"], [2, 2])
Sy.block[{}] = [[0, -0.5j], [0.5j, 0]]
Sz = CTensor(["I", "O"], [2, 2])
Sz.block[{}] = [[0.5, 0], [0, -0.5]]

SxSx = Sx.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sx.edge_rename({"I": "I1", "O": "O1"})).to(float)
SySy = Sy.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sy.edge_rename({"I": "I1", "O": "O1"})).to(float)
SzSz = Sz.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sz.edge_rename({"I": "I1", "O": "O1"})).to(float)

SS = SxSx + SySy + SzSz


class StateType(Enum):
    NotSet = 0
    Exact = 1
    WithoutEnvironment = 2
    WithEnvironment = 3


class SquareLattice:

    def __init__(self, M: int, N: int, D: int, Dv: int = 2) -> None:
        self.M: int = M
        self.N: int = N
        self.Dimension_Physics: int = D

        self.Dimension_Virtual: int = Dv

        self._state_type: StateType = StateType.NotSet

        self.vector: Tensor = Tensor()
        self.lattice: List[List[Tensor]] = []
        self.environment: Dict[Tuple[str, int, int], Tensor] = {}

        self.hamiltonian: Dict

        self.spin: List[List[int]] = []

        self._auxiliaries: Dict[Tuple[str, int, int]] = {}
        self._lattice_spin: List[List[Tensor]] = []

    @property
    def state_type(self):
        return self._state_type

    @state_type.setter
    def state_type(self, new_state_type):
        if new_state_type == StateType.NotSet:
            raise RuntimeError("Can not unset state type")

        if self._state_type == StateType.Exact:
            # 不可以从exact到另两个state type
            if new_state_type != StateType.Exact:
                raise RuntimeError("Cannot convert exact state to network state")

        elif self._state_type == StateType.WithoutEnvironment:
            if new_state_type == StateType.Exact:
                # WithoutEnv -> Exact
                self._contract_all_network()
                self.lattice = []

            elif new_state_type == StateType.WithEnvironment:
                # WithoutEnv -> WithEnv
                self._construct_environment()

        elif self._state_type == StateType.WithEnvironment:
            if new_state_type == StateType.Exact:
                # WithEnv -> Exact
                self._absorb_environment()
                self.environment = {}
                self._contract_all_network()
                self.lattice = []

            elif new_state_type == StateType.WithoutEnvironment:
                # WithEnv -> WithoutEnv
                self._absorb_environment()
                self.environment = {}

        else:
            self._initialize(new_state_type)

        self._state_type = new_state_type

    def _initialize(self, state_type):
        if state_type == StateType.Exact:
            self._initialize_vector()
        elif state_type == StateType.WithoutEnvironment:
            self._initialize_network()
        elif state_type == StateType.WithEnvironment:
            self._initialize_network()
            self._construct_environment()
        else:
            raise RuntimeError("State type not set")

    def _initialize_vector(self):
        name_list = [f"P-{i}-{j}" for i in range(self.M) for j in range(self.N)]
        dimension_list = [self.Dimension_Physics for _ in range(self.M) for _ in range(self.N)]
        self.vector = Tensor(name_list, dimension_list).randn()

    def _initialize_network(self):
        self.lattice = [[self._initialize_tensor_in_network(i, j) for j in range(self.N)] for i in range(self.M)]

    def _initialize_tensor_in_network(self, i, j):
        name_list = ["P"]
        dimension_list = [self.Dimension_Physics]
        if i != 0:
            name_list.append("U")
            dimension_list.append(self.Dimension_Virtual)
        if j != 0:
            name_list.append("L")
            dimension_list.append(self.Dimension_Virtual)
        if i != self.M - 1:
            name_list.append("D")
            dimension_list.append(self.Dimension_Virtual)
        if j != self.N - 1:
            name_list.append("R")
            dimension_list.append(self.Dimension_Virtual)
        return Tensor(name_list, dimension_list).randn()

    def _absorb_environment(self):
        for i in range(self.M):
            for j in range(self.N - 1):
                self.lattice[i][j] = self.lattice[i][j].multiple(self.environment[("R", i, j)], "U")

        for i in range(self.M - 1):
            for j in range(self.N):
                self.lattice[i][j] = self.lattice[i][j].multiple(self.environment[("D", i, j)], "U")

    def _construct_environment(self):
        for i in range(self.M):
            for j in range(self.N - 1):
                self.environment[("R", i, j)] = Tensor([",SVD_U", ",SVD_V"], [self.Dimension_Virtual, self.Dimension_Virtual]).identity({(",SVD_U", ",SVD_V")})

        for i in range(self.M - 1):
            for j in range(self.N):
                self.environment[("D", i, j)] = Tensor([",SVD_U", ",SVD_V"], [self.Dimension_Virtual, self.Dimension_Virtual]).identity({(",SVD_U", ",SVD_V")})

    def _contract_all_network(self):
        self.vector = Tensor(1)
        for i in range(self.M):
            for j in range(self.N):
                self.vector = self.vector.contract(self.lattice[i][j].edge_rename({"D": f"D-{j}", "P": f"P-{i}-{j}"}), {("R", "L"), (f"D-{j}", "U")})


lattice = SquareLattice(3, 3, 2)
lattice.state_type = StateType.WithoutEnvironment
lattice.state_type = StateType.Exact
print(lattice.vector)
