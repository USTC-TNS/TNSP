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

from copyreg import _slotnames
from ..abstract_state import AbstractState
from ..common_toolkit import send, allreduce_iterator_buffer, bcast_iterator_buffer, showln


class Configuration:
    """
    The configuration object for ansatz product state.
    """

    __slots__ = ["owner", "_configuration"]

    def export_orbit0(self):
        return self._configuration.export_orbit0()

    def copy(self):
        return Configuration(self.owner, self._configuration)

    def __init__(self, owner, config=None):
        """
        Create configuration for the given ansatz product state.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state owning this configuration.
        config : ConfigData, optional
            The preset configuration.
        """
        self.owner: AnsatzProductState = owner

        # Data storage of configuration, access it by configuration[l1, l2, orbit] instead
        if config is None:
            self._configuration = owner.Tensor.model.Configuration(owner.L1, owner.L2)
        else:
            self._configuration = config.copy()

    def __setitem__(self, key, value):
        self._configuration[key] = value

    def __getitem__(self, key):
        return self._configuration[key]

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
        return [[{orbit: self._configuration[l1, l2, orbit]
                  for orbit in self.owner.physics_edges[l1, l2]}
                 for l2 in range(self.owner.L2)]
                for l1 in range(self.owner.L1)]

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
                    self._configuration[l1, l2, orbit] = edge_point


class AnsatzProductState(AbstractState):
    """
    The ansatz product state, which is an expression of several subansatz.
    """

    __slots__ = ["ansatz"]

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
        # version 3 to version 4
        if state["data_version"] == 3:
            from .ansatzes.product_ansatz import ProductAnsatz
            state["ansatz"] = ProductAnsatz(self, state.pop("ansatzes"))
            state["data_version"] = 4
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

    def set_ansatz(self, ansatz, name):
        from .ansatzes.product_ansatz import ProductAnsatz
        self.ansatz = ProductAnsatz(self, {name: ansatz})

    def add_ansatz(self, ansatz, name):
        if self.ansatz is None:
            self.set_ansatz(ansatz, name)
            return
        from .ansatzes.sum_ansatz import SumAnsatz
        from .ansatzes.product_ansatz import ProductAnsatz
        if isinstance(self.ansatz, ProductAnsatz) or (isinstance(self.ansatz, SumAnsatz) and
                                                      len(self.ansatz.ansatzes) == 1):
            ansatzes = {}
            for key, value in zip(self.ansatz.names, self.ansatz.ansatzes):
                ansatzes[key] = value
            ansatzes[name] = ansatz
        else:
            ansatzes = {name: ansatz, "base": self.ansatz}
        self.ansatz = SumAnsatz(self, ansatzes)

    def mul_ansatz(self, ansatz, name):
        if self.ansatz is None:
            self.set_ansatz(ansatz, name)
            return
        from .ansatzes.sum_ansatz import SumAnsatz
        from .ansatzes.product_ansatz import ProductAnsatz
        if isinstance(self.ansatz, ProductAnsatz) or (isinstance(self.ansatz, SumAnsatz) and
                                                      len(self.ansatz.ansatzes) == 1):
            ansatzes = {}
            for key, value in zip(self.ansatz.names, self.ansatz.ansatzes):
                ansatzes[key] = value
            ansatzes[name] = ansatz
        else:
            ansatzes = {name: ansatz, "base": self.ansatz}
        self.ansatz = ProductAnsatz(self, ansatzes)

    def show_ansatz(self):
        showln(self.ansatz.show())

    def __init__(self, abstract, ansatz=None):
        """
        Create ansatz product state from a given abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create ansatz product state.
        ansatz : Ansatz, optional
            The ansatz that this ansatz product state using.
        """
        super()._init_by_copy(abstract)

        self.ansatz = ansatz

    def apply_gradient(self, gradient, step_size):
        """
        Apply the gradient to the state.

        Parameters
        ----------
        gradient : Tensors
            The gradient calculated by observer object.
        step_size : float
            The gradient step size.
        """
        setter = self.ansatz.tensors(None)
        setter.send(None)
        for tensor, grad in zip(self.ansatz.tensors(None), self.ansatz.tensors(gradient)):
            send(setter, tensor - grad * step_size)
        self.ansatz.refresh_auxiliaries()

    def state_conjugate(self, a=None):
        return self.ansatz.ansatz_conjugate(a)

    def state_prod_sum(self, a=None, b=None):
        return self.ansatz.ansatz_prod_sum(a, b)

    def state_dot(self, a=None, b=None):
        return self.ansatz.ansatz_dot(a, b)

    def allreduce_state(self, a=None):
        allreduce_iterator_buffer(self.ansatz.buffers(a))

    def bcast_state(self, a=None, root=0):
        bcast_iterator_buffer(self.ansatz.buffers(a), root=root)
