#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import sys
import numpy as np
import openfermion

file_name = sys.argv[1]
result_name = sys.argv[2]
data = openfermion.MolecularData(filename=file_name)
print("n_electrons", data.n_electrons, "n_qubits", data.n_qubits)
result = openfermion.transforms.get_fermion_operator(data.get_molecular_hamiltonian())
np.save(result_name, result)
