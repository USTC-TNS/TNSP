#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

# set random seed
seed(2333)

# what you can create is lattice for simple update only
# different model have different parameter for create
su_create("kitaev", L1=2, L2=2, D=4, Jx=1, Jy=1, Jz=1)

# save or open file
su_dump("/dev/null")
# su_load xxx

# total_step, step_size, new_dimension
su_update(400, 0.01, 5)
# for system size > 4*4, it is dangerous to get exact state
su_to_ex()
ex_energy()
ex_update(1000, 4)
ex_energy()

su_to_gm()
gm_run(100,
       10,
       0.01,
       configuration_cut_dimension=8,
       use_natural_gradient=True,
       use_line_search=True,
       log_file="run.log",
       measurement="kitaev.Sz")
