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

from copyreg import _slotnames
import numpy as np
import torch
from ..abstract_state import AbstractState
from ..tensor_element import tensor_element
from ..utility import bcast_buffer


def index_tensor_element(tensor, pool={}):
    """
    This function convert the hamiltonian elements in TAT tensor format to simple format, which flatten all fermion
    operators, with the specific fermion operators order.
    """
    tensor_id = id(tensor)
    if tensor_id not in pool:
        rank = tensor.rank // 2
        standard_names = [f"O{i}" for i in range(rank)] + list(reversed([f"I{i}" for i in range(rank)]))
        element_pool = tensor_element(tensor)
        result = {}
        for x, pack in element_pool.items():
            new_x = tuple(tensor.edge_by_name(f"O{rank}").index_by_point(config) for rank, config in enumerate(x))
            tensor_new_x = torch.tensor(new_x, device=torch.device("cpu"), dtype=torch.int64)
            new_pack = result[new_x] = {}
            for y, tensor_shrinked in pack.items():
                new_y = tuple(tensor.edge_by_name(f"O{rank}").index_by_point(config) for rank, config in enumerate(y))
                tensor_new_y = torch.tensor(new_y, device=torch.device("cpu"), dtype=torch.int64)
                new_pack[new_y] = (
                    tensor_shrinked.transpose(standard_names).storage[0].item(),
                    tensor_new_x,
                    tensor_new_y,
                )
        pool[tensor_id] = result
    return pool[tensor_id]


class torch_grad:

    def __init__(self, enable_grad):
        self.prev = torch.is_grad_enabled()
        self.enable_grad = enable_grad

    def __enter__(self):
        if self.enable_grad is not None:
            torch.set_grad_enabled(self.enable_grad)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False


class Configuration:
    """
    Configuration for sampling neural state.
    This class only works for exportinng and importing,
    which has the same interface with configuration of sampling lattice.
    """

    __slots__ = ["owner", "_data"]

    def copy(self, _=None):
        """
        Copy the configuration.

        Returns
        -------
        Configuration
            The new configuration.
        """
        result = Configuration(owner)
        result._data = self._data.copy()
        return result

    def __init__(self, owner, _=None):
        """
        Create configuration for the given sampling neural state.

        Parameters
        ----------
        owner : SamplingNeuralState
            The sampling neural state owning this configuration.
        """
        self.owner = owner
        max_orbit = max(orbit for [l1, l2, orbit], _ in self.owner.physics_edges)
        # -1 : not exist, -2 : not set
        self._data = np.zeros([self.owner.L1, self.owner.L2, max_orbit + 1], dtype=np.int64) - 1
        for [l1, l2, orbit], edge in self.owner.physics_edges:
            self._data[l1, l2, orbit] = -2

    def __getitem__(self, l1l2o):
        """
        Get the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.

        Returns
        -------
        EdgePoint | None
            The configuration of the specific site.
        """
        index = self._data[l1l2o]
        if index == -2 or index == -1:
            return None
        else:
            return self.owner.physics_edges[l1l2o].point_by_index(index)

    def __setitem__(self, l1l2o, value):
        """
        Set the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.
        value : ?EdgePoint
            The configuration of this site.
        """
        if value is None:
            index = -2
        else:
            value = self._construct_edge_point(value)
            index = self.owner.physics_edges[l1l2o].index_by_point(value)
        self._data[l1l2o] = index

    def __delitem__(self, l1l2o):
        """
        Clear the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.
        """
        self.__setitem__(l1l2o, None)

    def export_configuration(self):
        """
        Export the configuration of all the sites.

        Returns
        -------
        Tensor
            The result configuration in numpy format
        """
        return self._data.copy()

    def import_configuration(self, config):
        """
        Import the configuration of all the sites.

        Parameters
        ----------
        config : Tensor
            The imported configuration in numpy format

        Returns
        -------
        Self
            This function will return the instance itself.
        """
        self._data = config.copy()
        return self

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


class SamplingNeuralState(AbstractState):
    """
    neural state used for sampling.
    """

    __slots__ = ["network", "_op_pool"]

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
            state["data_version"] = 3
        # version 3 to version 4
        if state["data_version"] == 3:
            state["data_version"] = 4
        # version 4 to version 5
        if state["data_version"] == 4:
            state["data_version"] = 5
        # version 5 to version 6
        if state["data_version"] == 5:
            state["data_version"] = 6
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        del state["_op_pool"]  # _op_pool is cache, should not be dumped.
        return state

    def __init__(self, abstract):
        """
        Create a sampling neural state from abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create sampling neural state.
        """
        super()._init_by_copy(abstract)

        self.network = None

    @property
    def op_pool(self):
        """
        The pool to cache whether an index of a specified edge is fermionic.
        """
        if not hasattr(self, "_op_pool"):
            max_orbit_index = max(orbit for [l1, l2, orbit], edge in self.physics_edges)
            max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in self.physics_edges)
            result = torch.zeros(
                [self.L1, self.L2, max_orbit_index + 1, max_physical_dim],
                dtype=torch.bool,
                device=torch.device("cpu"),  # It should be placed in cpu since we will check it frequently later
            )
            flag = False
            for [l1, l2, orbit], edge in self.physics_edges:
                for dim in range(edge.dimension):
                    sym, _ = edge.point_by_index(dim)
                    if sym.parity:
                        flag = True
                        result[l1, l2, orbit, dim] = True
            if flag:
                self._op_pool = result
            else:
                self._op_pool = None
        return self._op_pool

    @property
    def dtype(self):
        """
        Get the dtype of the network.

        Returns
        -------
        torch.dtype
            The dtype of the amplitude.
        """
        dtype = next(self.network.parameters()).dtype
        if self.Tensor.is_complex:
            dtype = dtype.to_complex()
        return dtype

    @property
    def device(self):
        """
        Get the device of the network.

        Returns
        -------
        torch.device
            The device where the current network is placed
        """
        return next(self.network.parameters()).device

    def __call__(self, configurations, enable_grad=None):
        """
        Get the amplitude of the given configurations

        Parameters
        ----------
        configurations : Tensor
            The list of configurations in pytorch format.
        enable_grad : bool, optional
            Whether to explicit enable torch autograd.

        Returns
        -------
        Tensor
            The list of the result amplitudes in pytorch format.
        """
        with torch_grad(enable_grad):
            return self.network(configurations)

    def holes(self, value):
        """
        Calculate (partial value / partial network) / value

        Parameters
        ----------
        value : Tensor
            The pytorch scalar which will be called with function backward.

        Returns
        -------
        Tensor
            The gradient over value in 1d torch tensor format.
        """
        if self.Tensor.is_complex:
            with torch_grad(True):
                value.real.backward(retain_graph=True)
            real = torch.cat([param.grad.reshape([-1]) for param in self.network.parameters() if param.requires_grad])
            self.network.zero_grad()
            with torch_grad(True):
                value.imag.backward()
            imag = torch.cat([param.grad.reshape([-1]) for param in self.network.parameters() if param.requires_grad])
            self.network.zero_grad()
            result = (real + 1j * imag)
        else:
            value.backward()
            result = torch.cat([param.grad.reshape([-1]) for param in self.network.parameters() if param.requires_grad])
            self.network.zero_grad()
        result = result / value
        return result.detach_()

    def set_gradient(self, grad):
        """
        Set the gradient to the state for optimizer to use.

        Parameters
        ----------
        grad : Tensor
            The gradient calculated by observer object.
        """
        index = 0
        for tensor in self.network.parameters():
            if tensor.requires_grad:
                size = tensor.nelement()
                if tensor.grad is None:
                    tensor.grad = torch.zeros_like(tensor)
                tensor.grad += grad[index:index + size].reshape(tensor.shape)
                index += size

    def apply_gradient(self, grad, step):
        """
        Apply the gradient to the state.

        Parameters
        ----------
        grad : Tensor
            The gradient calculated by observer object.
        step : float
            The gradient step size.
        """
        with torch_grad(False):
            index = 0
            for tensor in self.network.parameters():
                if tensor.requires_grad:
                    size = tensor.nelement()
                    tensor -= step * grad[index:index + size].reshape(tensor.shape)
                    index += size

    def state_dot(self, a=None, b=None):
        """
        Calculate the dot of two state shape data. If None is given, its own state data will be used.

        Parameters
        ----------
        a, b : Tensor | None
            The lattice shape data.
        """
        if a is None:
            a = self.state_vector()
        if b is None:
            b = self.state_vector()
        return torch.dot(a.conj(), b).real

    def state_vector(self):
        """
        Get the state data in 1d torch tensor format.

        Returns
        -------
        Tensor
            The state data in 1d torch tensor format.
        """
        result = torch.cat([param.reshape([-1]) for param in self.network.parameters() if param.requires_grad])
        return result.detach_()

    def bcast_state(self):
        """
        Bcast the state across MPI processes.
        """
        state = self.state_vector()
        bcast_buffer(state)

        index = 0
        for tensor in self.network.parameters():
            size = tensor.nelement()
            tensor.data = state[index:index + size].reshape(tensor.shape)
            index += size
