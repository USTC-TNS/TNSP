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

import os
import sys
try:
    import cPickle as pickle
except:
    import pickle
import signal
import importlib
from mpi4py import MPI
import numpy as np
import TAT

clear_line = "\u001b[2K"

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def show(*args, **kwargs):
    if mpi_rank == 0:
        print(clear_line, *args, **kwargs, end="\r")


def showln(*args, **kwargs):
    if mpi_rank == 0:
        print(clear_line, *args, **kwargs)


def allreduce_buffer(buffer):
    mpi_comm.Allreduce(MPI.IN_PLACE, buffer)


def allreduce_iterator_buffer(iterator):
    requests = []
    for tensor in iterator:
        requests.append(mpi_comm.Iallreduce(MPI.IN_PLACE, tensor))
    MPI.Request.Waitall(requests)


def allreduce_lattice_buffer(lattice):
    return allreduce_iterator_buffer(tensor.storage for row in lattice for tensor in row)


def bcast_buffer(buffer, root=0):
    mpi_comm.Bcast(buffer, root=root)


def bcast_iterator_buffer(iterator, root=0):
    requests = []
    for tensor in iterator:
        requests.append(mpi_comm.Ibcast(tensor, root=root))
    MPI.Request.Waitall(requests)


def bcast_lattice_buffer(lattice, root=0):
    return bcast_iterator_buffer((tensor.storage for row in lattice for tensor in row), root=root)


class SignalHandler():

    __slots__ = ["signal", "sigint_recv", "saved_handler"]

    def __init__(self, handler_signal):
        self.signal = handler_signal
        self.sigint_recv = 0
        self.saved_handler = None

    def __enter__(self):

        def handler(signum, frame):
            print(f"\n process {mpi_rank} receive {self.signal.name}, send again to send {self.signal.name}\u001b[2F")
            if self.sigint_recv == 1:
                self.saved_handler(signum, frame)
            else:
                self.sigint_recv = 1

        self.saved_handler = signal.signal(self.signal, handler)
        return self

    def __call__(self):
        if self.sigint_recv:
            print(f" process {mpi_rank} receive {self.signal.name}")
        result = mpi_comm.allreduce(self.sigint_recv)
        self.sigint_recv = 0
        return result != 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        signal.signal(self.signal, self.saved_handler)


class SeedDiffer:

    __slots__ = ["seed"]

    max_int = 2**31
    random_int = TAT.random.uniform_int(0, max_int - 1)

    def make_seed_diff(self):
        self.seed = (self.random_int() + mpi_rank) % self.max_int
        TAT.random.seed(self.seed)
        # c++ random engine will generate the same first uniform int if the seed is near.
        TAT.random.uniform_real(0, 1)()

    def make_seed_same(self):
        self.seed = mpi_comm.allreduce(self.random_int() // mpi_size)
        TAT.random.seed(self.seed)

    def __enter__(self):
        self.make_seed_diff()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.make_seed_same()

    def __init__(self):
        self.make_seed_same()


seed_differ = SeedDiffer()


def read_from_file(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)


def write_to_file(obj, file_name):
    if mpi_rank == 0:
        tmp_file_name = f".{file_name}.tmp"
        with open(tmp_file_name, "wb") as file:
            pickle.dump(obj, file)
        os.rename(tmp_file_name, file_name)
    mpi_comm.barrier()


@np.vectorize
def lattice_conjugate(tensor):
    return tensor.conjugate(default_is_physics_edge=True)


@np.vectorize
def lattice_dot(tensor_1, tensor_2):
    return tensor_1.contract(tensor_2, {(name, name) for name in tensor_1.names}).storage[0]


def lattice_prod_sum(tensors_1, tensors_2):
    dot = lattice_dot(tensors_1, tensors_2)
    return np.sum(dot)


def lattice_update(tensors_1, tensors_2):
    L1, L2 = tensors_1.shape
    for l1 in range(L1):
        for l2 in range(L2):
            tensors_1[l1, l2] += tensors_2[l1, l2]


@np.vectorize
def lattice_randomize(tensor):
    random_same_shape = tensor.same_shape().rand(0, 1)
    random_same_shape.storage *= np.sign(tensor.storage)
    return random_same_shape


def import_from_tetpath(full_name):
    names = full_name.split(".")
    length = len(names)
    if "TETPATH" in os.environ:
        path = os.environ["TETPATH"].split(":")
    else:
        path = []
    for i in range(length):
        name = ".".join(names[:i + 1])
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            raise ModuleNotFoundError(f"No module named '{full_name}'")
        path = spec.submodule_search_locations
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def get_imported_function(module_name_or_function, function_name):
    if isinstance(module_name_or_function, str):
        # 1. current folder
        # 2. TETPATH
        # 3. tetraku
        # 4. normal import
        try:
            module = import_from_tetpath(module_name_or_function)
        except ModuleNotFoundError:
            try:
                module = importlib.import_module("." + module_name_or_function, "tetraku.models")
            except ModuleNotFoundError:
                try:
                    module = importlib.import_module("." + module_name_or_function, "tetraku.ansatzes")
                except ModuleNotFoundError:
                    module = importlib.import_module(module_name_or_function)
        return getattr(module, function_name)
    else:
        return module_name_or_function


def send(receiver, value):
    try:
        receiver.send(value)
    except StopIteration:
        pass


def safe_contract(tensor_1, tensor_2, pair, *, contract_all_physics_edges=False):
    new_pair = set()
    if contract_all_physics_edges:
        for name in tensor_1.names:
            if str(name)[0] == "P" and name in tensor_2.names:
                new_pair.add((name, name))
    for name_1, name_2 in pair:
        if name_1 in tensor_1.names and name_2 in tensor_2.names:
            new_pair.add((name_1, name_2))

    return tensor_1.contract(tensor_2, new_pair)


def safe_rename(tensor, name_map):
    new_name_map = {}
    for key, value in name_map.items():
        if key in tensor.names:
            new_name_map[key] = value
    return tensor.edge_rename(new_name_map)
