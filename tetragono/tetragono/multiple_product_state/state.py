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
from ..common_toolkit import send


class MultipleProductState(AbstractState):
    """
    The multiple product state, which is product of several subansatz.
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
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

    def __init__(self, abstract):
        """
        Create multiple product state from a given abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create multiple product state.
        """
        super()._init_by_copy(abstract)
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
        configurations : list[list[list[dict[int, EdgePoint]]]]
            The given configurations to calculate weight and delta.
        calculate_delta : set[str]
            The iterator of name of ansatz to calculate delta.

        Returns
        -------
        tuple[list[complex | float], list[dict[str, ansatz]]]
            The weight and the delta ansatz.
        """
        number = len(configurations)
        weight = [1. for _ in range(number)]
        delta = [{} for _ in range(number)]
        for name, ansatz in self.ansatzes.items():
            sub_weight, sub_delta = ansatz.weight_and_delta(configurations, name in calculate_delta)
            for i in range(number):
                weight[i] *= sub_weight[i]
            if sub_delta is not None:
                for i in range(number):
                    delta[i][name] = sub_delta[i] / sub_weight[i]
        for i in range(number):
            this_weight = weight[i]
            this_delta = delta[i]
            for name in this_delta:
                this_delta[name] *= this_weight
        return weight, delta

    def get_norm_max(self, delta, names):
        """
        Get the max norm of the delta or state.

        Parameters
        ----------
        delta : None | dict[str, Delta]
            The delta or state to calculate.
        names : None | list[str]
            The ansatzes to calculate norm.

        Returns
        -------
            The max norm.
        """
        result = 0.0
        for name in names:
            ansatz = self.ansatzes[name]
            if delta is None:
                this = ansatz.get_norm_max(None)
            else:
                this = ansatz.get_norm_max(delta[name])
            if this > result:
                result = this
        return result

    def fix_relative_to_state(self, gradient):
        """
        Get fixed relative to state for a state shape data.

        Parameters
        ----------
        gradient : dict[str, Delta]
            The state shape data.

        Returns
        -------
        dict[str, Delta]
            The result state shape data which have the same norm to the state itself.
        """
        param = self.get_norm_max(None, gradient.keys()) / self.get_norm_max(gradient, gradient.keys())
        return gradient * param

    def apply_gradient(self, gradient, step_size):
        """
        Apply the gradient to the state.

        Parameters
        ----------
        gradient : dict[str, Delta]
            The gradient calculated by observer object.
        step_size : float
            The gradient step size.
        """
        for name, _ in gradient.items():
            setter = self.ansatzes[name].buffers(None)
            setter.send(None)
            for tensor, grad in zip(self.ansatzes[name].buffers(None), self.ansatzes[name].buffers(gradient[name])):
                send(setter, tensor - grad * step_size)
        self.refresh_auxiliaries()

    def refresh_auxiliaries(self):
        """
        Refresh auxiliaries after updating state.
        """
        for name in self.ansatzes:
            self.ansatzes[name].refresh_auxiliaries()
