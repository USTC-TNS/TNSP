#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from weakref import ref
from itertools import chain


class Copy:
    """
    A helper class used to copy entire lazy node graph.

    To copy a lazy node graph totolly, create a copy object first, `copier = Copy()`. Then copy node one by one in the
    correct order with respect to the dependence relationship by `new_node = copier(old_node)`.
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

        This function will replace the old upstream node to the new upstream node correctly based on the data stored in
        this copy object. So after copying, there should be no relation between old graph and new graph, only if the
        entire graph is copied completely.

        Parameters
        ----------
        node : Node[T]
            The old lazy node from the old graph.

        Returns
        -------
        Node[T]
            The newly created lazy node in the new graph.
        """
        # If the object is copied before, return the result node directly from the database stored in this copy object.
        if node in self._map:
            return self._map[node]

        # Get the args and kwargs, the only different between old one and new one is replacing old upstream to new
        # upstream, where new upstream was created by call this copy object before, if not, the old node and new node
        # will share the same upstream.
        args = tuple(self._map_node(i) for i in node._args)  # map all args of node
        kwargs = {i: self._map_node(j) for i, j in node._kwargs.items()}  # Map all kwargs of node

        # Create new node
        result = Node(node._func, *args, **kwargs)

        # Update cache of the newly created node, if all upstream did not change, the new node will use the old cache.
        cache_valid = True
        # Check the difference one by one in all the upstream of this node.
        for new, old in chain(zip(result._args, node._args), zip(result._kwargs.values(), node._kwargs.values())):
            # Only check if the args or kwargs is also a node.
            if isinstance(old, Node):
                # If old upstream and new upstream own different cache, the cache is invalid.
                if old._value != new._value:
                    cache_valid = False
                    break
        # All upstream cache did not change, so the newly created node also keep the cache same to old node.
        if cache_valid:
            result._value = node._value

        # Record relation in the database stored in this copy object.
        self._map[node] = result

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
            If the parameter `node` is really a lazy node and it is recorded in `self._map`, it will return newly
            created mapped by the input node. Otherwise it will return the input object directly without any change.
        """
        # map old upstream node object to new upstream, it must be a lazy node if mapped, since only node object will be
        # put into self._map. But it may be out of the map even if it is a lazy node, then old node and new node will
        # share the same upstream.

        # If it is in the map, just map it, if not, keep it unchanged.
        if node in self._map:
            return self._map[node]
        else:
            return node


class Node:
    """
    Lazy node type, used to build a lazy evaluation graph, the value of it may be reset and when trying to get value of
    it, the needed node will be calculated automatically.
    """

    __slots__ = ["_value", "_downstream", "_func", "_args", "_kwargs", "__weakref__"]

    def _clear_downstream_of_upstream(self):
        """
        Clear itself from the downstream list of all its upstreams. It is called when deleting this node.
        """
        # Loop over all its args and kwargs.
        for i in chain(self._args, self._kwargs.values()):
            # If it is a node, then it is an upstream.
            if isinstance(i, Node):
                # So remove it.
                i._downstream.remove(ref(self))

    def _add_downstream_of_upstream(self):
        """
        Add itself to the downstream list of all its upstreams. It is called when creating this node.
        """
        # Loop over all its args and kwargs.
        for i in chain(self._args, self._kwargs.values()):
            # If it is a node, then it is an upstream.
            if isinstance(i, Node):
                # So add it.
                i._downstream.add(ref(self))

    def __del__(self):
        """
        The destructor of lazy node.
        """
        self._clear_downstream_of_upstream()

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

        # Add itsself to the downstream list of all its upstreams.
        self._add_downstream_of_upstream()

    def reset(self, value=None):
        """
        Reset value of this node, it will refresh all its downstream. If the value is given, this node will be reset by
        this value.

        Parameters
        ----------
        value : T | None, default=None
            Reset the cache of this node by the given value.
        """
        # If the value does not change, it reset nothing, so do nothing.
        if self._value != value:
            # Update the cache of this node.
            self._value = value
            # Reset all its downstream
            for i in self._downstream:
                # All its downstream should be valid, since when a downstream deleted, the downstream will remove itself
                # from the downstream list of this node.
                i().reset()

    def __bool__(self):
        """
        Check if the value of this node is already calculated.

        Returns
        -------
        bool
            Return True if the value is already calculated, otherwise return False.
        """
        return self._value is not None

    def _make_frame(self):
        """
        Make a frame for the calculation of this node to be pushed into the stack.

        The data of a frame contains node, args, kwargs, keys and index.
        The keys is keys list of kwargs, it is to ensure the order of the loop.
        The index contains four part, current frame pointer, size of args, size of kwargs and size of args and kwargs.

        Returns
        -------
        tuple[Node, args, kwargs, list[key], int]
            The frame used for calculating the value of this node.
        """
        l_args = len(self._args)
        l_kwargs = len(self._kwargs)
        index = [0, l_args, l_kwargs, l_args + l_kwargs]
        return (self, [*self._args], {**self._kwargs}, tuple(self._kwargs.keys()), index)

    def __call__(self):
        """
        Obtain the value of this node, it will calculate the value by self._func and cache it.

        Returns
        -------
        T
            The calculated value of this node.
        """
        # If the cache is empty, calculate this node and update the cache.
        if self._value is None:
            # Create an new stack and simulate recursion by it.
            stack = [self._make_frame()]
            self._recursion(stack)
        # Return its cache.
        return self._value

    @staticmethod
    def _recursion(stack):
        """
        Run recursion on lazy graph.

        Parameters
        ----------
        stack : list[tuple[Node, args, kwargs, list[key], int]]
            The stack used to run recursion.
        """
        # If the stack is empty, all works done, return
        while len(stack) != 0:
            # Calculate the top of the stack
            node, args, kwargs, keys, index = stack[-1]
            # Loop over args and kwargs
            for i in range(index[0], index[3]):
                # Get the arg of this pointer
                if i < index[1]:
                    arg = args[i]
                else:
                    key = keys[i - index[1]]
                    arg = kwargs[key]
                # If it is node, it is needed to get its value
                if isinstance(arg, Node):
                    if arg._value is not None:
                        # If the upstream own cache, just use it
                        if i < index[1]:
                            args[i] = arg._value
                        else:
                            kwargs[key] = arg._value
                    else:
                        # Otherwise push it to the top of the stack and continue the outer loop
                        stack.append(arg._make_frame())
                        # Update current frame pointer to avoid useless checking when the flow come back into this frame
                        index[0] = i
                        break
            else:
                # All args and kwargs have been calculated, calculate its own value of this node
                node._value = node._func(*args, **kwargs)
                # And pop this node calulcation frame.
                stack.pop()


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
    # Change a node with an empty returned function.
    result = Node(lambda: None)
    # And reset its value forcely, since getting value of a node will check cache first, it will skip calling that
    # invalid empty returned function.
    result.reset(value)

    return result
