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

from ..common_variable import mpi_comm


def expand_sampling_lattice_dimension(state, new_dimension, epsilon):
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l1 != 0 and l1 % 2 == 0:
                expand_vertical(state, l1 - 1, l2, new_dimension, epsilon)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l1 != 0 and l1 % 2 == 1:
                expand_vertical(state, l1 - 1, l2, new_dimension, epsilon)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l2 != 0 and l2 % 2 == 0:
                expand_horizontal(state, l1, l2 - 1, new_dimension, epsilon)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l2 != 0 and l2 % 2 == 1:
                expand_horizontal(state, l1, l2 - 1, new_dimension, epsilon)

    for l1 in range(state.L1):
        for l2 in range(state.L2):
            state[l1, l2] = mpi_comm.bcast(state[l1, l2], root=0)


def expand_horizontal(state, l1, l2, new_dimension, epsilon):
    left = state[l1, l2]
    right = state[l1, l2 + 1]
    left_q, left_r = left.qr("r", {*(p_name for p_name in left.names if p_name[0] == "P"), "R"}, "R", "L")
    right_q, right_r = right.qr("r", {*(p_name for p_name in left.names if p_name[0] == "P"), "L"}, "L", "R")
    left_r = left_r.edge_rename({name: f"L_{name}" for name in left_r.names})
    right_r = right_r.edge_rename({name: f"R_{name}" for name in right_r.names})
    big = left_r.contract(right_r, {("L_R", "R_L")})
    norm = big.norm_max()
    big += big.same_shape().randn() * epsilon * norm
    u, s, v = big.svd({l_name for l_name in big.names if l_name[:2] == "L_"}, "R", "L", "L", "R", new_dimension)
    left = left_q.contract(u, {("R", "L_L")}).contract(s, {("R", "L")})
    right = right_q.contract(v, {("L", "R_R")})
    state[l1, l2] = left.edge_rename({l_name: l_name[2:] for l_name in left.names if l_name[:2] == "L_"})
    state[l1, l2 + 1] = right.edge_rename({r_name: r_name[2:] for r_name in right.names if r_name[:2] == "R_"})


def expand_vertical(state, l1, l2, new_dimension, epsilon):
    up = state[l1, l2]
    down = state[l1 + 1, l2]
    up_q, up_r = up.qr("r", {*(p_name for p_name in up.names if p_name[0] == "P"), "D"}, "D", "U")
    down_q, down_r = down.qr("r", {*(p_name for p_name in down.names if p_name[0] == "P"), "U"}, "U", "D")
    up_r = up_r.edge_rename({name: f"U_{name}" for name in up_r.names})
    down_r = down_r.edge_rename({name: f"D_{name}" for name in down_r.names})
    big = up_r.contract(down_r, {("U_D", "D_U")})
    norm = big.norm_max()
    big += big.same_shape().randn() * epsilon * norm
    u, s, v = big.svd({u_name for u_name in big.names if u_name[:2] == "U_"}, "D", "U", "U", "D", new_dimension)
    up = up_q.contract(u, {("D", "U_U")}).contract(s, {("D", "U")})
    down = down_q.contract(v, {("U", "D_D")})
    state[l1, l2] = up.edge_rename({u_name: u_name[2:] for u_name in up.names if u_name[:2] == "U_"})
    state[l1 + 1, l2] = down.edge_rename({d_name: d_name[2:] for d_name in down.names if d_name[:2] == "D_"})
