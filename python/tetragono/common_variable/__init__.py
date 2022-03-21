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

import signal
from mpi4py import MPI
import TAT
from . import No
from . import Fermi
from . import FermiU1_tJ
from . import Fermi_Hubbard

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


def allreduce_lattice_buffer(lattice):
    requests = []
    for row in lattice:
        for tensor in row:
            requests.append(mpi_comm.Iallreduce(MPI.IN_PLACE, tensor.storage))
    MPI.Request.Waitall(requests)


def bcast_buffer(buffer, root):
    mpi_comm.Bcast(buffer, root=root)


def bcast_lattice_buffer(lattice, root):
    requests = []
    for row in lattice:
        for tensor in row:
            requests.append(mpi_comm.Ibcast(tensor.storage, root=root))
    MPI.Request.Waitall(requests)


class SignalHandler():

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
        signal.signal(self.signal, self.saved_handler)


class SeedDiffer:
    max_int = 2**31
    random_int = TAT.random.uniform_int(0, max_int - 1)

    def make_seed_diff(self):
        TAT.random.seed((self.random_int() + mpi_rank) % self.max_int)
        # c++ random engine will generate the same first uniform int if the seed is near.
        TAT.random.uniform_real(0, 1)()

    def make_seed_same(self):
        TAT.random.seed(mpi_comm.allreduce(self.random_int() // mpi_size))

    def __enter__(self):
        self.make_seed_diff()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.make_seed_same()

    def __init__(self):
        self.make_seed_same()


seed_differ = SeedDiffer()
