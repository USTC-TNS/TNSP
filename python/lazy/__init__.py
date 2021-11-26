#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import weakref


class Copy:
    """
    A helper class used to copy entire lazy node graph.
    """

    __slots__ = ["_map"]

    def __init__(self):
        """
        Create a new Copy class used to copy entire lazy node graph.
        """
        self._map = {}  # A map from old lazy node to new lazy node

    def __call__(self, node):
        """
        Copy a lazy node into newly created graph and get result node.

        Parameters
        ----------
        node : Node[T]
            The old lazy node from the old graph.

        Returns
        -------
        Node[T]
            The newly created lazy node in the new graph.
        """
        args = tuple(self._map_node(i) for i in node._args)  # map all args of node
        kwargs = {i: self._map_node(j) for i, j in node._kwargs.items()}  # Map all kwargs of node
        result = Node(node._func, *args, **kwargs)
        result._value = node._value  # Set value
        self._map[node] = result  # Record relation in map
        return result

    def _map_node(self, node):
        """
        Get new created lazy node from old lazy node.

        Parameters
        ----------
        node : Node[T] | Any
            The old lazy node from the old graph, or any other things.

        Returns
        -------
        Node[T] | Any
            If the parameter `node` is really a lazy node, it will return newly created mapped by the input node.
            Otherwise it will return the input object directly without any change.
        """
        if isinstance(node, Node):
            return self._map[node]
        else:
            return node


class Node:
    """
    Lazy node type, used to build a lazy evaluation graph, the value of it may be reset and when trying to get value of
    it, the needed node will be calculated automatically.
    """

    __slots__ = ["_value", "_downstream", "_func", "_args", "_kwargs", "__weakref__"]

    def __init__(self, func, *args, **kwargs):
        """
        Create a lazy node with given function and arguments.

        Parameters
        ----------
        func : Callable[..., T]
            The function used to calculate the value of this node.
        *args, **kwargs
            Arguments used by the function, if there is lazy node inside args, it will be calculated first when try to
            calculate this lazy node.
        """
        self._value = None  # The cache of the value of this lazy node
        self._downstream = set()  # The set of weak reference of all downstream node
        self._func = func
        self._args = args
        self._kwargs = kwargs

        for i in self._args:
            if isinstance(i, Node):
                i._downstream.add(weakref.ref(self))

    def reset(self, value=None):
        """
        Reset value of this node, it will refresh all its downstream.

        Parameters
        ----------
        value : T | None, default=None
            Reset the cache of this node by the given value.
        """
        if self._value != value:
            self._value = value
            new_downstream = set()
            for i in self._downstream:
                if i():
                    i().reset()
                    new_downstream.add(i)

            self._downstream = new_downstream

    def __call__(self):
        """
        Obtain the value of this node, it will calculate the value by self._func and cache it

        Returns
        -------
        T
            The calculated value of this node.
        """
        if self._value is None:
            # Do NOT use generator here for less stack depth

            # args = [self._unwrap_node(i) for i in self._args]
            args = []
            for i in self._args:
                if isinstance(i, Node):
                    i = i()
                args.append(i)

            # kwargs = {i: self._unwrap_node(j) for i, j in self._kwargs.items()}
            kwargs = {}
            for i, j in self._kwargs.items():
                if isinstance(j, Node):
                    j = j()
                kwargs[i] = j

            self._value = self._func(*args, **kwargs)
        return self._value


def Root(value=None):
    """
    Create a lazy node with given value.

    Parameters
    ----------
    value : T | None, default=None
        The given value of this lazy node.

    Returns
    -------
    Node[T]
        The result lazy node.
    """
    result = Node(lambda: None)
    result.reset(value)
    return result
