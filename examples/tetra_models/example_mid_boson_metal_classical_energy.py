#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from tetragono.shell import *

seed(2333)

gm_create("boson_metal", L1=10, L2=10, D=4, J=-1, K=3, mu=-1.6)


def classical_energy(configuration):
    n_up = 0
    for l1 in range(configuration._owner.L1):
        for l2 in range(configuration._owner.L2):
            if configuration[l1, l2, 0][1] == 0:
                n_up += 1
    return (n_up - 37)**2


gm_run(
    1000,
    100,
    0.01,
    configuration_cut_dimension=12,
    classical_energy=classical_energy,
    use_natural_gradient=True,
    momentum_parameter=0.9,
    use_fix_relative_step_size=True,
    log_file="run.log",
)
