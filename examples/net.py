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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Set, Tuple, TypeVar

VertexType = TypeVar("VertexType")
EdgeType = TypeVar("EdgeType")


@dataclass
class Vertex(Generic[VertexType]):
    information: Optional[VertexType]
    neighbor_names: Set[str]


@dataclass
class Edge(Generic[EdgeType]):
    information: Optional[EdgeType]
    first_name: str
    second_name: str


class Graph(Generic[VertexType, EdgeType]):

    def __init__(self):
        self.vertices: Dict[str, Vertex[VertexType]] = {}
        self.edges: Dict[str, Edge[EdgeType]] = {}

    def has_vertex(self, vertex_name: str) -> bool:
        return vertex_name in self.vertices

    def has_edge(self, edge_name: str) -> bool:
        return edge_name in self.edges

    def add_vertex(self, vertex_name: str, information: Optional[VertexType] = None):
        if self.has_vertex(vertex_name):
            raise KeyError("Vertex Already Exists")
        else:
            self.vertices[vertex_name] = Vertex[VertexType](information, set())

    def add_edge(self, edge_name: str, vertex_1_name: str, vertex_2_name: str, information: Optional[EdgeType] = None):
        if self.has_edge(edge_name):
            raise KeyError("Edge Already Exists")
        else:
            self.edges[edge_name] = Edge[EdgeType](information, vertex_1_name, vertex_2_name)
            self.vertices[vertex_1_name].neighbor_names.add(edge_name)
            self.vertices[vertex_2_name].neighbor_names.add(edge_name)

    def add_half_edge(self, edge_name: str, vertex_edge: str, information: Optional[EdgeType] = None, put_vertex_first: bool = True):
        if self.has_edge(edge_name):
            raise KeyError("Edge Already Exists")
        else:
            if put_vertex_first:
                self.edges[edge_name] = Edge[EdgeType](information, vertex_name, "")
            else:
                self.edges[edge_name] = Edge[EdgeType](information, "", vertex_name)
            self.vertices[vertex_name].neighbor_names.add(edge_name)

    def set_vertex(self, vertex_name: str, information: Optional[VertexType]) -> Optional[VertexType]:
        vertex_information = self.vertices[vertex_name].information
        self.vertices[vertex_name].information = information
        return vertex_information

    def set_edge(self, edge_name: str, information: Optional[EdgeType]) -> Optional[EdgeType]:
        edge_information = self.edges[edge_name].information
        self.edges[edge_name].information = information
        return edge_information

    def remove_vertex(self, vertex_name: str) -> Optional[VertexType]:
        for edge_name in self.vertices[vertex_name].neighbor_names:
            if self.edges[edge_name].first == name:
                self.edges[edge_name].first = ""
            else:
                self.edges[edge_name].second = ""
        vertex_information = self.vertices[vertex_name].information
        del self.vertices[vertex_name]
        return vertex_information

    def remove_edge(self, edge_name: str) -> Optional[EdgeType]:
        edge_information = self.edges[edge_name].information
        vertex_1_name = self.edges[edge_name].first_name
        vertex_2_name = self.edges[edge_name].second_name
        if vertex_1_name != "":
            self.vertices[vertex_1_name].neighbor_names.remove(edge_name)
        if vertex_2_name != "":
            self.vertices[vertex_2_name].neighbor_names.remove(edge_name)
        del self.edges[edge_name]
        return edge_information

    def rename_vertex(self, old_name: str, new_name: str) -> None:
        vertex = self.vertices[old_name]
        for edge_name in vertex.neighbor_names:
            if self.edges[edge_name].first_name == old_name:
                self.edges[edge_name].first_name = new_name
            if self.edges[edge_name].second_name == old_name:
                self.edges[edge_name].second_name = new_name
        del self.vertices[old_name]
        self.vertices[new_name] = vertex

    def rename_edge(self, old_name: str, new_name: str) -> None:
        edge = self.edges[old_name]
        vertex_1 = self.vertices[edge.first_name]
        vertex_1.neighbor_names.remove(old_name)
        vertex_1.neighbor_names.add(new_name)
        vertex_2 = self.vertices[edge.second_name]
        vertex_2.neighbor_names.remove(old_name)
        vertex_2.neighbor_names.add(new_name)
        del self.edges[old_name]
        self.edges[new_name] = edge

    def split_edge(self, name):
        # TODO
        pass

    def absorb(self, name_1: str, name_2: str, function: Callable[[Any, Any, Set[Tuple[str, Any]]], Any]):
        vertex_1 = self.vertices[name_1].information
        vertex_2 = self.vertices[name_2].information
        # find all edge connect vertex_1 and vertex_2, NOTICE: no direction here
        edge_set: Set[Tuple[str, Any]] = set()
        for name, [edge_information, vertex_1_name, vertex_2_name] in self.edges:
            if {vertex_1_name, vertex_2_name} == {name_1, name_2}:
                edge_set.add((name, edge_information))
        # absorb
        new_vertex_information = function(vertex_1, vertex_2, edge_set)
        # update structure

    def exude(self, name_1, name_2):
        pass


"""
import TAT


class TensorNetwork(Graph):
    Tensor = TAT(float)
    add_site = Graph.add_vertex
    remove_site = Graph.remove_vertex
    add_bond = Graph.add_edge
    remove_bond = Graph.remove_edge


Tensor = TAT.Tensor.DNo


def main():
    net = TensorNetwork()
    net.add_site("A", Tensor(2))
    net.add_site("B", Tensor(3))
    net.add_bond("AB", "A", "B")
    # net.absorb("A", "B", net.contract)


if __name__ == "__main__":
    main()
"""
