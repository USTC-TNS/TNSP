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

from . import No
from . import Fermi
from . import FermiU1_tJ
from . import Fermi_Hubbard

clear_line = "\u001b[2K"

from mpi4py import MPI

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
