#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import numpy as np
from tetragono.common_tensor.tensor_toolkit import kronecker_product, rename_io
from tetragono.common_tensor.Fermi import CP, CM, I
from tetragono.common_tensor.Fermi_Hubbard import put_sign_in_H

# merge order: up down
CPU = kronecker_product(rename_io(CP, [0]), rename_io(I, [1])).merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"]
}, put_sign_in_H, {"O0"})
CPD = kronecker_product(rename_io(CP, [1]), rename_io(I, [0])).merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"]
}, put_sign_in_H, {"O0"})
CMU = kronecker_product(rename_io(CM, [0]), rename_io(I, [1])).merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"]
}, put_sign_in_H, {"O0"})
CMD = kronecker_product(rename_io(CM, [1]), rename_io(I, [0])).merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"]
}, put_sign_in_H, {"O0"})

CMUD = kronecker_product(rename_io(CMU, [0]).edge_rename({"T": "T0"}), rename_io(CMD, [1]).edge_rename({"T": "T1"}))
CMDU = kronecker_product(rename_io(CMD, [0]).edge_rename({"T": "T0"}), rename_io(CMU, [1]).edge_rename({"T": "T1"}))
CPUD = kronecker_product(rename_io(CPU, [0]).edge_rename({"T": "T0"}), rename_io(CPD, [1]).edge_rename({"T": "T1"}))
CPDU = kronecker_product(rename_io(CPD, [0]).edge_rename({"T": "T0"}), rename_io(CPU, [1]).edge_rename({"T": "T1"}))

# CPAB CMCD will give (c_0A c_1B)^dagger (c_0C c_1D)
# CPAB.contract(CMCD, {("T0","T0"), ("T1","T1")})
CM_singlet = (CMUD - CMDU) / np.sqrt(2)
CP_singlet = (CPUD - CPDU) / np.sqrt(2)
CM_triplet = (CMUD + CMDU) / np.sqrt(2)
CP_triplet = (CPUD + CPDU) / np.sqrt(2)

singlet = rename_io(CP_singlet, [0, 1]).contract(rename_io(CM_singlet, [2, 3]), {("T0", "T0"), ("T1", "T1")}).to(float)

triplet = rename_io(CP_triplet, [0, 1]).contract(rename_io(CM_triplet, [2, 3]), {("T0", "T0"), ("T1", "T1")}).to(float)

singlet_pool = {(): singlet}
triplet_pool = {(): triplet}


# link = ((p, m),...)
# where p=0,1 and m=2,3
# Ip and Om will be linked
def get_singlet(*link):
    if link not in singlet_pool:
        result = singlet.trace({(f"I{p}", f"O{m}") for p, m in link})
        result = result.edge_rename({f"I{m}": f"I{p}" for p, m in link})
        if "I3" in result.names and "I2" not in result.names:
            result = rename_io(result, {3: 2})
        singlet_pool[link] = result
    return singlet_pool[link]


def get_triplet(*link):
    if link not in triplet_pool:
        result = triplet.trace({(f"I{p}", f"O{m}") for p, m in link})
        result = result.edge_rename({f"I{m}": f"I{p}" for p, m in link})
        if "I3" in result.names and "I2" not in result.names:
            result = rename_io(result, {3: 2})
        triplet_pool[link] = result
    return triplet_pool[link]
