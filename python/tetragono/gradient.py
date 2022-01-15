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

from __future__ import annotations
import pickle
import signal
import TAT
from .sampling_lattice import SamplingLattice, DirectSampling, SweepSampling, ErgodicSampling, Observer
from .common_variable import show, showln, mpi_comm, mpi_rank, mpi_size


class SignalHandler():

    def __init__(self, signal):
        self.signal = signal
        self.sigint_recv = 0
        self.saved_handler = None

    def __enter__(self):

        def handler(signum, frame):
            if self.sigint_recv == 1:
                self.saved_handler(signum, frame)
            else:
                self.sigint_recv = 1

        self.saved_handler = signal.signal(self.signal, handler)
        return self

    def __call__(self):
        if self.sigint_recv:
            print(f" process {mpi_rank} receive {self.signal.name}")
        result = mpi_comm.allreduce(self.sigint_recv)
        self.sigint_recv = 0
        return result != 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(self.signal, self.saved_handler)


class SeedDiffer:
    max_int = 2**31
    random_int = TAT.random.uniform_int(0, max_int - 1)

    def make_seed_diff(self):
        TAT.random.seed((self.random_int() + mpi_rank) % self.max_int)

    def make_seed_same(self):
        TAT.random.seed(mpi_comm.allreduce(self.random_int() // mpi_size))

    def __enter__(self):
        self.make_seed_diff()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.make_seed_same()

    def __init__(self):
        self.make_seed_same()


seed_differ = SeedDiffer()


def check_difference(state, observer, grad, reweight_observer, configuration_pool, check_difference_delta):

    def get_energy():
        with reweight_observer:
            for possibility, configuration in configuration_pool:
                configuration.refresh_all()
                reweight_observer(possibility, configuration)
        return reweight_observer.energy[0] * state.site_number

    original_energy = observer.energy[0] * state.site_number
    delta = check_difference_delta
    showln(f"difference delta is hard coded as {delta}")
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            showln(l1, l2)
            s = state[l1, l2].storage
            g = grad[l1][l2].conjugate(positive_contract=True).storage
            for i in range(len(s)):
                s[i] += delta
                now_energy = get_energy()
                rgrad = (now_energy - original_energy) / delta
                s[i] -= delta
                if state.Tensor.is_complex:
                    s[i] += delta * 1j
                    now_energy = get_energy()
                    igrad = (now_energy - original_energy) / delta
                    s[i] -= delta * 1j
                    cgrad = rgrad + igrad * 1j
                else:
                    cgrad = rgrad
                showln(" ", cgrad, g[i])


def line_search(state, observer, grad, reweight_observer, configuration_pool, step_size, param, line_search_amplitude,
                line_search_error_threshold):
    saved_state = [[state[l1, l2] for l2 in range(state.L2)] for l1 in range(state.L1)]
    grad_dot_pool = {}

    def grad_dot(eta):
        if eta not in grad_dot_pool:
            for l1 in range(state.L1):
                for l2 in range(state.L2):
                    state[l1, l2] = saved_state[l1][l2] - eta * param * grad[l1][l2].conjugate(positive_contract=True)
            with reweight_observer:
                for possibility, configuration in configuration_pool:
                    configuration.refresh_all()
                    reweight_observer(possibility, configuration)
                    show(f"predicting eta={eta}, energy={reweight_observer.energy}")
            result = mpi_comm.bcast(observer._lattice_dot(grad, reweight_observer.gradient), root=0)
            showln(f"predict eta={eta}, energy={reweight_observer.energy}, gradient dot={result}")
            grad_dot_pool[eta] = result
        return grad_dot_pool[eta]

    grad_dot_pool[0] = mpi_comm.bcast(observer._lattice_dot(grad, observer.gradient), root=0)
    if grad_dot(0.0) > 0:
        begin = 0.0
        end = step_size * line_search_amplitude

        if grad_dot(end) > 0:
            step_size = end
            showln(f"step_size is chosen as {step_size}, since grad_dot(begin) > 0, grad_dot(end) > 0")
        else:
            while True:
                x = (begin + end) / 2
                if grad_dot(x) > 0:
                    begin = x
                else:
                    end = x
                if (end - begin) / end < line_search_error_threshold:
                    step_size = begin
                    showln(f"step_size is chosen as {step_size}, since step size error < {line_search_error_threshold}")
                    break
    else:
        showln(f"step_size is chosen as {step_size}, since grad_dot(begin) < 0")
        step_size = step_size

    for l1 in range(state.L1):
        for l2 in range(state.L2):
            state[l1, l2] = saved_state[l1][l2]
    return mpi_comm.bcast(step_size, root=0)


def gradient_descent(
        state: SamplingLattice,
        sampling_total_step=0,
        grad_total_step=1,
        grad_step_size=0,
        *,
        # About sampling
        sampling_method="direct",
        configuration_cut_dimension=None,
        direct_sampling_cut_dimension=4,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=20,
        metric_inverse_epsilon=0.01,
        sj_shift_per_site=None,
        # About gradient method
        use_check_difference=False,
        use_line_search=False,
        use_fix_relative_step_size=False,
        # About log and save state
        log_file=None,
        save_state_interval=None,
        save_state_file=None,
        # About line search
        line_search_amplitude=1.25,
        line_search_error_threshold=0.1,
        # About check difference
        check_difference_delta=1e-8):

    # Gradient step
    use_gradient = grad_step_size != 0 or use_check_difference
    if use_gradient:
        if use_check_difference:
            grad_total_step = 1
        else:
            grad_total_step = grad_total_step
    else:
        grad_total_step = 1
    showln(f"gradient total step={grad_total_step}")

    # Prepare observers
    observer = Observer(state)
    observer.add_energy()
    if use_gradient:
        showln("calculate gradient")
        observer.enable_gradient()
        if use_natural_gradient:
            observer.enable_natural_gradient()
        need_reweight_observer = use_line_search or use_check_difference
        if need_reweight_observer:
            reweight_observer = Observer(state)
            reweight_observer.add_energy()
            reweight_observer.enable_gradient()
    else:
        showln("do NOT calculate gradient")

    # Sampling method
    if sampling_method == "sweep":
        showln("using sweep sampling")
        # Use direct sampling to find sweep sampling initial configuration.
        sampling = DirectSampling(state, configuration_cut_dimension, direct_sampling_cut_dimension)
        sampling()
        configuration = sampling.configuration
        sampling = SweepSampling(state, configuration_cut_dimension)
        sampling.configuration = configuration
        sampling_total_step = sampling_total_step
    elif sampling_method == "ergodic":
        showln("using ergodic sampling")
        sampling = ErgodicSampling(state, configuration_cut_dimension)
        sampling_total_step = sampling.total_step
    elif sampling_method == "direct":
        showln("using direct sampling")
        sampling = DirectSampling(state, configuration_cut_dimension, direct_sampling_cut_dimension)
        sampling_total_step = sampling_total_step
    else:
        raise ValueError("Invalid sampling method")

    # Main loop
    with SignalHandler(signal.SIGINT) as sigint_handler:
        for grad_step in range(grad_total_step):
            if need_reweight_observer:
                configuration_pool = []
            # Sampling and observe
            with seed_differ, observer:
                for sampling_step in range(sampling_total_step):
                    if sampling_step % mpi_size == mpi_rank:
                        possibility, configuration = sampling()
                        observer(possibility, configuration)
                        if need_reweight_observer:
                            configuration_pool.append((possibility, configuration.copy()))
                        show(
                            f"sampling, total_step={sampling_total_step}, energy={observer.energy}, step={sampling_step}"
                        )
            showln(f"sampling done, total_step={sampling_total_step}, energy={observer.energy}")
            if use_gradient:
                # Save log
                if log_file and mpi_rank == 0:
                    with open(log_file, "a") as file:
                        print(*observer.energy, file=file)

                # Get gradient
                if use_natural_gradient:
                    show("calculating natural gradient")
                    grad = observer.natural_gradient(conjugate_gradient_method_step,
                                                     metric_inverse_epsilon,
                                                     sj_shift_per_site=sj_shift_per_site)
                    showln("calculate natural gradient done")
                else:
                    grad = observer.gradient

                # Change state
                if use_check_difference:
                    showln("checking difference")
                    check_difference(state, observer, grad, reweight_observer, configuration_pool,
                                     check_difference_delta)

                elif use_line_search:
                    showln("line searching")
                    param = mpi_comm.bcast((observer._lattice_dot(state._lattice, state._lattice) /
                                            observer._lattice_dot(grad, grad))**0.5,
                                           root=0)
                    grad_step_size = line_search(state, observer, grad, reweight_observer, configuration_pool,
                                                 grad_step_size, param, line_search_amplitude,
                                                 line_search_error_threshold)
                    real_step_size = grad_step_size * param
                    for l1 in range(state.L1):
                        for l2 in range(state.L2):
                            state[l1, l2] -= real_step_size * grad[l1][l2].conjugate(positive_contract=True)
                elif use_fix_relative_step_size:
                    showln("fix relative step size")
                    param = mpi_comm.bcast((observer._lattice_dot(state._lattice, state._lattice) /
                                            observer._lattice_dot(grad, grad))**0.5,
                                           root=0)
                    real_step_size = grad_step_size * param
                    for l1 in range(state.L1):
                        for l2 in range(state.L2):
                            state[l1, l2] -= real_step_size * grad[l1][l2].conjugate(positive_contract=True)
                else:
                    real_step_size = grad_step_size
                    for l1 in range(state.L1):
                        for l2 in range(state.L2):
                            state[l1, l2] -= real_step_size * grad[l1][l2].conjugate(positive_contract=True)
                showln(f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}")

                # Bcast state and refresh sampling(refresh sampling aux and sampling config)
                for l1 in range(state.L1):
                    for l2 in range(state.L2):
                        state[l1, l2] = mpi_comm.bcast(state[l1, l2], root=0)
                sampling.refresh_all()

                # Save state
                if save_state_interval and (grad_step + 1) % save_state_interval == 0:
                    if save_state_file and mpi_rank == 0:
                        with open(save_state_file, "wb") as file:
                            pickle.dump(state, file)
            if sigint_handler():
                break
