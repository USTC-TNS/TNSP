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
from typing import TypeVar, Generic
import weakref
from multimethod import multimethod

__all__ = ["Copier", "Node", "Root"]


class Copier:

    def __init__(self):
        self._map = {}

    def __call__(self, node):
        result = Node(node, self)
        self._map[node] = result
        return result

    def _map_node(self, node):
        if isinstance(node, Node):
            return self._map[node]
        else:
            return node


T = TypeVar('T')


class Node(Generic[T]):

    @multimethod
    def __init__(self, other: Node[T], copier: Copier):
        self._value: T | None = other._value
        self._downstream: set[Node] = set()
        self._func = other._func
        self._args = [copier._map_node(i) for i in other._args]
        self._kwargs = {i: copier._map_node(j) for i, j in other._kwargs.items()}

        for i in self._args:
            if isinstance(i, Node):
                i._downstream.add(weakref.ref(self))

    @multimethod
    def __init__(self, func, *args, **kwargs):
        self._value: T | None = None
        self._downstream: set[Node] = set()
        self._func = func
        self._args = args
        self._kwargs = kwargs

        for i in self._args:
            if isinstance(i, Node):
                i._downstream.add(weakref.ref(self))

    def reset(self, value: T | None = None):
        if self._value != value:
            self._value = value
            new_downstream: set[Node] = set()
            for i in self._downstream:
                if i():
                    i().reset()
                    new_downstream.add(i)

            self._downstream = new_downstream

    def __call__(self) -> T:
        if self._value is None:
            args = [self._unwrap_node(i) for i in self._args]
            kwargs = {i: self._unwrap_node(j) for i, j in self._kwargs.items()}
            self._value = self._func(*args, **kwargs)
        return self._value

    @staticmethod
    def _unwrap_node(node):
        if isinstance(node, Node):
            return node()
        else:
            return node


def Root(value: T | None = None) -> Node[T]:
    result = Node(lambda: None)
    result.reset(value)
    return result
