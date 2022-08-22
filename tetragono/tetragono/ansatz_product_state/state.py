#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import itertools
from copyreg import _slotnames
import numpy as np
from ..abstract_state import AbstractState
from ..common_toolkit import send, showln, allreduce_iterator_buffer, bcast_iterator_buffer


class Configuration:
    """
    The configuration object for ansatz product state.
    """

    __slots__ = ["owner", "_configuration"]

    def __init__(self, owner, config=None):
        """
        Create configuration for the given ansatz product state.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state owning this configuration.
        config : list[list[dict[int, ?EdgePoint]]], optional
            The preset configuration.
        """
        self.owner: AnsatzProductState = owner

        # Data storage of configuration, access it by configuration[l1, l2, orbit] instead
        self._configuration = [[{orbit: None
                                 for orbit in self.owner.physics_edges[l1, l2]}
                                for l2 in range(self.owner.L2)]
                               for l1 in range(self.owner.L1)]

        if config is not None:
            self.import_configuration(config)

    def _construct_edge_point(self, value):
        """
        Construct edge point from something that can be used to construct an edge point.

        Parameters
        ----------
        value : ?EdgePoint
            Edge point or something that can be used to construct a edge point.

        Returns
        -------
        EdgePoint
            The result edge point object.
        """
        if not isinstance(value, tuple):
            symmetry = self.owner.Symmetry()  # work for NoSymmetry
            index = value
        else:
            symmetry, index = value
        symmetry = self.owner._construct_symmetry(symmetry)
        return (symmetry, index)

    def __setitem__(self, key, value):
        l1, l2, orbit = key
        value = self._construct_edge_point(value)
        if self._configuration[l1][l2][orbit] != value:
            self._configuration[l1][l2][orbit] = value

    def __getitem__(self, key):
        l1, l2, orbit = key
        return self._configuration[l1][l2][orbit]

    def __delitem__(self, key):
        self.__setitem__(key, None)

    def export_configuration(self):
        """
        Export the configuration of all the sites.

        Returns
        -------
        list[list[dict[int, EdgePoint]]]
            The configuration data of all the sites
        """
        return [[self._configuration[l1][l2].copy() for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]

    def import_configuration(self, config):
        """
        Import the configuration of all the sites.

        Parameters
        ----------
        config : list[list[dict[int, ?EdgePoint]]]
            The configuration data of all the sites
        """
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge_point in config[l1][l2].items():
                    self[l1, l2, orbit] = edge_point


class AnsatzProductState(AbstractState):
    """
    The ansatz product state, which is product of several subansatz.
    """

    __slots__ = ["ansatzes"]

    def __setstate__(self, state):
        # before data_version mechanism, state is (None, state)
        if isinstance(state, tuple):
            state = state[1]
        # before data_version mechanism, there is no data_version field
        if "data_version" not in state:
            state["data_version"] = 0
        # version 0 to version 1
        if state["data_version"] == 0:
            state["data_version"] = 1
        # version 1 to version 2
        if state["data_version"] == 1:
            state["data_version"] = 2
        # version 2 to version 3
        if state["data_version"] == 2:
            self._v2_to_v3_rename(state)
            state["data_version"] = 3
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

    def __init__(self, abstract):
        """
        Create ansatz product state from a given abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create ansatz product state.
        """
        super()._init_by_copy(abstract)

        # A dict from ansatz name to ansatz object
        self.ansatzes = {}

    def add_ansatz(self, ansatz, name=None):
        """
        Add an ansatz.

        Parameters
        ----------
        ansatz : Ansatz
            The ansatz to be made.
        name : str, optional
            The name of the new ansatz.
        """
        if name is None:
            name = str(len(self.ansatzes))
        self.ansatzes[name] = ansatz

    def weight_and_delta(self, configurations, calculate_delta):
        """
        Calculate weight and delta of all ansatz.

        Parameters
        ----------
        configurations : list[Configuration]
            The given configurations to calculate weight and delta.
        calculate_delta : list[str]
            The list of name of ansatz to calculate delta.

        Returns
        -------
        tuple[list[complex | float], list[list[Delta]]]
            The weight and the delta ansatz, where The weight part is weight[config_index], and the delta part is
            delta[config_index][ansatz_index].
        """
        number = len(configurations)
        weight = [1. for _ in range(number)]
        delta = [[None for _ in calculate_delta] for _ in range(number)]
        for name, ansatz in self.ansatzes.items():
            sub_weight, sub_delta = ansatz.weight_and_delta(configurations, name in calculate_delta)
            for i in range(number):
                weight[i] *= sub_weight[i]
            if sub_delta is not None:
                ansatz_index = calculate_delta.index(name)
                for i in range(number):
                    delta[i][ansatz_index] = sub_delta[i] / sub_weight[i]
        for i in range(number):
            this_weight = weight[i]
            this_delta = delta[i]
            for j, _ in enumerate(this_delta):
                this_delta[j] *= this_weight
        return weight, delta

    def apply_gradient(self, gradient, step_size, *, part):
        """
        Apply the gradient to the state.

        Parameters
        ----------
        gradient : list[Delta]
            The gradient calculated by observer object.
        step_size : float
            The gradient step size.
        part : list[str]
            The ansatzes to be updated
        """
        for i, name in enumerate(part):
            setter = self.ansatzes[name].buffers(None)
            setter.send(None)
            for tensor, grad in zip(self.ansatzes[name].buffers(None), self.ansatzes[name].buffers(gradient[i])):
                send(setter, tensor - grad * step_size)
        self.refresh_auxiliaries()

    def state_prod_sum(self, a=None, b=None, *, part):
        """
        Calculate the summary of product of two state like data, only calculate the dot of some ansatz based on part
        variable.

        Parameters
        ----------
        a, b : list[Delta], optional
            The two state like data, if not given, the data the state itself stored will be used.
        part : list[str]
            The ansatzes to calculate.
        """
        if a is None:
            a = [None for name in part]
        if b is None:
            b = [None for name in part]
        result = 0.0
        for i, name in enumerate(part):
            result += self.ansatzes[name].ansatz_prod_sum(a[i], b[i])
        return result

    def state_conjugate(self, a=None, *, part):
        """
        Calculate the conjugate of the given state like data, only calculate the ansatz set in part variable.

        Parameters
        ----------
        a : list[Delta], optional
            The state like data, if not given, the data the state itself stored will be used.
        part : list[str]
            The ansatzes to calculate.

        Returns
        -------
        list[Delta]
            The conjugate result.
        """
        if a is None:
            a = [None for name in part]
        return np.array([self.ansatzes[name].ansatz_conjugate(a[i]) for i, name in enumerate(part)], dtype=object)

    def state_dot(self, a=None, b=None, *, part):
        """
        Calculate the dot product of two state like data, only calculate the dot of some ansatz based on part variable.

        Parameters
        ----------
        a, b : list[Delta], optional
            The two state like data, if not given, the data the state itself stored will be used.
        part : list[str]
            The ansatzes to calculate.
        """
        if a is None:
            a = [None for name in part]
        if b is None:
            b = [None for name in part]
        result = 0.0
        for i, name in enumerate(part):
            result += self.ansatzes[name].ansatz_dot(a[i], b[i])
        return result

    def allreduce_state(self, a=None, *, part):
        if a is None:
            a = [None for name in part]
        allreduce_iterator_buffer(
            itertools.chain(*(self.ansatzes[name].buffers_for_mpi(a[i]) for i, name in enumerate(part))))

    def bcast_state(self, a=None, root=0, *, part):
        if a is None:
            a = [None for name in part]
        bcast_iterator_buffer(
            itertools.chain(*(self.ansatzes[name].buffers_for_mpi(a[i]) for i, name in enumerate(part))), root=root)

    def refresh_auxiliaries(self):
        """
        Refresh auxiliaries after updating state.
        """
        for name in self.ansatzes:
            self.ansatzes[name].refresh_auxiliaries()

    def normalize_state(self):
        """
        Normalize the state
        """
        for name in self.ansatzes:
            self.ansatzes[name].normalize_ansatz()
