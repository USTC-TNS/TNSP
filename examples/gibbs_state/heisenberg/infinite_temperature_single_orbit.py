#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import TAT
import tetragono as tet


def initialize_infinite_temperature(lattice, sigma=0):
    for row in lattice._lattice:
        for tensor in row:
            tensor.randn(0, sigma)
            position = {name: 0 for name in tensor.names}
            tensor[position] = 1
            position["P0"] = position["P1"] = 1
            tensor[position] = 1
    return lattice


def main(argv):
    file_name = argv[1]
    sigma = float(argv[2])
    seed = int(argv[3])
    TAT.random.seed(seed)
    lattice = tet.read_from_file(file_name)
    lattice = initialize_infinite_temperature(lattice, sigma=sigma)
    tet.write_to_file(lattice, file_name)


if __name__ == "__main__":
    main(sys.argv)
