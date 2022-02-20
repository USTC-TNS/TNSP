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

import TAT

Tensor = TAT.No.Z.Tensor

identity = Tensor(["I0", "O0"], [2, 2])
identity.blocks[identity.names] = [[1, 0], [0, 1]]

pauli_x = Tensor(["I0", "O0"], [2, 2])
pauli_x.blocks[pauli_x.names] = [[0, 1], [1, 0]]
Sx = pauli_x / 2

pauli_y = Tensor(["I0", "O0"], [2, 2])
pauli_y.blocks[pauli_y.names] = [[0, -1j], [1j, 0]]
Sy = pauli_y / 2

pauli_z = Tensor(["I0", "O0"], [2, 2])
pauli_z.blocks[pauli_z.names] = [[1, 0], [0, -1]]
Sz = pauli_z / 2

pauli_x_pauli_x = pauli_x.edge_rename({"I0": "I1", "O0": "O1"}).contract(pauli_x, set())
SxSx = Sx.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sx, set())
pauli_y_pauli_y = pauli_y.edge_rename({"I0": "I1", "O0": "O1"}).contract(pauli_y, set())
SySy = Sy.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sy, set())
pauli_z_pauli_z = pauli_z.edge_rename({"I0": "I1", "O0": "O1"}).contract(pauli_z, set())
SzSz = Sz.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sz, set())

SS = SxSx + SySy + SzSz
