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


def initialize_infinite_temperature(lattice):
    for row in lattice._lattice:
        for tensor in row:
            tensor.zero()
            position = {name: (0, 0) for name in tensor.names}
            tensor[position] = 1
            position["P0"] = (+1, 0)
            position["P1"] = (-1, 0)
            tensor[position] = 1
            position["P0"] = (+1, 1)
            position["P1"] = (-1, 1)
            tensor[position] = 1
            position["P0"] = (+2, 0)
            position["P1"] = (-2, 0)
            tensor[position] = 1
    return lattice


def main(argv):
    file_name = argv[1]
    lattice = tet.read_from_file(file_name)
    lattice = initialize_infinite_temperature(lattice)
    tet.write_to_file(lattice, file_name)


if __name__ == "__main__":
    main(sys.argv)
