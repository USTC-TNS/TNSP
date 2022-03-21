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
from ..sampling_tools import DirectSampling
from ..common_variable import mpi_comm, mpi_size, mpi_rank, show, showln, seed_differ


def plus_log(loga, logb):
    return loga + np.log(1 + np.exp(logb - loga))


def normalize_state(state, sampling_total_step, configuration_cut_dimension, direct_sampling_cut_dimension):
    sampling = DirectSampling(state, configuration_cut_dimension, None, direct_sampling_cut_dimension)
    log_prod_ws = 0.0
    with seed_differ:
        for sampling_step in range(sampling_total_step):
            if sampling_step % mpi_size == mpi_rank:
                possibility, configuration = sampling()
                log_prod_ws += np.log(np.abs(complex(configuration.hole(())).real))
                show(f"normalizing, total_step={sampling_total_step}, step={sampling_step}")
    log_prod_ws = mpi_comm.allreduce(log_prod_ws)
    showln(f"normalizing done, total_step={sampling_total_step}")
    param = np.exp((log_prod_ws / sampling_total_step) / (state.L1 * state.L2))
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            state[l1, l2] /= param
