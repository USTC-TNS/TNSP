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
from typing import TypeVar, Generic, Callable
import weakref

__all__ = ["Copier", "Merger", "Node", "Root"]


class Copier:
    __slots__ = ["_map"]

    def __init__(self) -> None:
        self._map: dict[Node, Node] = {}

    def __call__(self, node: Node) -> Node:
        result: Node = Node._copy(node, self)
        result._value = node._value
        self._map[node] = result
        return result

    def _map_node(self, node):
        if isinstance(node, Node):
            return self._map[node]
        else:
            return node


class Merger:
    __slots__ = ["_map", "_select_1", "_select_2"]

    def __init__(self) -> None:
        self._map: dict[Node, Node] = {}
        self._select_1: dict[Node] = set()  # nodes using value from channel 1
        self._select_2: dict[Node] = set()  # nodes using value from channel 2

    def __call__(self, node_1: Node, node_2: Node, select: int) -> Node:
        if select == 1:
            result: Node = Node._copy(node_1, self)
            result._value = node_1._value
            self._select_1(result)
        elif select == 2:
            result: Node = Node._copy(node_2, self)
            result._value = node_2._value
            self._select_2(result)
        elif select == 0:
            result: Node = Node._copy(node_1, self)
            select_1: bool = True
            select_2: bool = True
            upstreams: set[Node] = {i for i in result._args if isinstance(i, Node)} | {j for i, j in result._kwargs.items() if isinstance(j, Node)}
            for i in upstreams:
                if i not in self._select_1:
                    select_1 = False  # upstream not using channel 1 so result cannot use it
                if i not in self._select_2:
                    select_2 = False  # upstream not using channel 2 so result cannot use it
            if select_1:
                result._value = node_1._value
                self._select_1.add(result)
            elif select_2:
                result._value = node_2._value
                self._select_2.add(result)
            else:
                result._value = None
        else:
            raise ValueError("Invalid select index")
        self._map[node_1] = result
        self._map[node_2] = result
        return result

    def _map_node(self, node):
        if isinstance(node, Node):
            return self._map[node]
        else:
            return node


T = TypeVar('T')


class Node(Generic[T]):
    __slots__ = ["_value", "_downstream", "_func", "_args", "_kwargs", "__weakref__"]

    @staticmethod
    def _copy(other: Node[T], copier: Copier | Merger) -> Node[T]:
        args = [copier._map_node(i) for i in other._args]
        kwargs = {i: copier._map_node(j) for i, j in other._kwargs.items()}
        return Node(other._func, *args, **kwargs)

    def __init__(self, func: Callable[..., T], *args, **kwargs):
        self._value: T | None = None
        self._downstream: set[weakref.ref[Node]] = set()
        self._func: [..., T] = func
        self._args = args
        self._kwargs = kwargs

        for i in self._args:
            if isinstance(i, Node):
                i._downstream.add(weakref.ref(self))

    def reset(self, value: T | None = None):
        if self._value != value:
            self._value = value
            new_downstream: set[weakref.ref[Node]] = set()
            for i in self._downstream:
                if i():
                    i().reset()
                    new_downstream.add(i)

            self._downstream = new_downstream

    def __call__(self) -> T:
        if self._value is None:
            # need to reduce stack depth
            # or call sys.setrecursionlimit(more_stack_depth)

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


def Root(value: T | None = None) -> Node[T]:
    result = Node(lambda: None)
    result.reset(value)
    return result
