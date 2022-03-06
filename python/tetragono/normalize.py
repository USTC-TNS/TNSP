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
from .sampling_lattice import DirectSampling
from .gradient import seed_differ
from .common_variable import mpi_comm, mpi_size, mpi_rank, show, showln


def plus_log(loga, logb):
    return loga + np.log(1 + np.exp(logb - loga))


def normalize_state(state, sampling_total_step, configuration_cut_dimension, direct_sampling_cut_dimension):
    sampling = DirectSampling(state, configuration_cut_dimension, None, direct_sampling_cut_dimension)
    log_total_weight = 0.  # weight need to -= mpi_size at last
    count = 0
    with seed_differ:
        for sampling_step in range(sampling_total_step):
            if sampling_step % mpi_size == mpi_rank:
                possibility, configuration = sampling()
                log_reweight = np.log(abs(complex(configuration.hole(())).real)) * 2 - np.log(possibility)
                log_total_weight = plus_log(log_total_weight, log_reweight)
                count += 1
                show(f"normalizing, total_step={sampling_total_step}, step={sampling_step}")
    log_total_weight = mpi_comm.allreduce(log_total_weight, plus_log)
    log_total_weight += np.log(1 - np.exp(np.log(mpi_size) - log_total_weight)) - np.log(sampling_total_step)
    showln(f"normalizing done, total_step={sampling_total_step}, log<psi|psi>={log_total_weight}")
    param = np.exp(log_total_weight / (state.L1 * state.L2 * 2))
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            state[l1, l2] /= param
