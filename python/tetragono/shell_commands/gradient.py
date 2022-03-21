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

import pickle
import importlib
import signal
from datetime import datetime
import numpy as np
import TAT
from ..sampling_lattice import SamplingLattice
from ..sampling_tools import Observer, SweepSampling, ErgodicSampling, DirectSampling
from ..common_variable import show, showln, mpi_comm, mpi_rank, mpi_size, bcast_lattice_buffer, SignalHandler, seed_differ
from .fix_gauge_sampling import fix_sampling_lattice_guage


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


def try_normalize(state, mean_log_ws):
    param = np.exp(mean_log_ws / (state.L1 * state.L2))
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            state[l1, l2] /= param


def gradient_descent(
        state: SamplingLattice,
        sampling_total_step=0,
        grad_total_step=1,
        grad_step_size=0,
        *,
        # About observer
        cache_configuration=False,
        # About sampling
        sampling_method="direct",
        configuration_cut_dimension=None,
        direct_sampling_cut_dimension=4,
        sweep_initial_configuration=None,
        configuration_dump_file=None,
        # About subspace
        restrict_subspace=None,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=20,
        metric_inverse_epsilon=0.01,
        sj_shift_per_site=None,
        # About gradient method
        use_check_difference=False,
        use_line_search=False,
        use_fix_relative_step_size=False,
        use_random_gradient=False,
        momentum_parameter=0.0,
        # About gauge fixing
        fix_gauge=False,
        # About log and save state
        log_file=None,
        save_state_interval=None,
        save_state_file=None,
        # About line search
        line_search_amplitude=1.25,
        line_search_error_threshold=0.1,
        # About momentum
        orthogonalize_momentum=False,
        # About check difference
        check_difference_delta=1e-8,
        # About Measurement
        measurement=None):

    time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

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
    if sj_shift_per_site is not None:
        cache_configuration = True
        showln("using shaojun's method, cache configuration")

    # Restrict subspace
    if restrict_subspace is not None:
        restrict = importlib.import_module(restrict_subspace).restrict
    else:
        restrict = None

    # Prepare observers
    observer = Observer(state, restrict)
    observer.add_energy()
    if measurement:
        measurement_modules = {}
        measurement_names = measurement.split(",")
        for measurement_name in measurement_names:
            measurement_module = importlib.import_module(measurement_name)
            measurement_modules[measurement_name] = measurement_module
            observer.add_observer(measurement_name, measurement_module.measurement(state))
    if cache_configuration:
        observer.cache_configuration()
    if use_gradient:
        showln("calculate gradient")
        observer.enable_gradient()
        if use_natural_gradient:
            observer.enable_natural_gradient()
        need_reweight_observer = use_line_search or use_check_difference
    else:
        showln("do NOT calculate gradient")
        need_reweight_observer = False
    if need_reweight_observer:
        reweight_observer = Observer(state, restrict)
        reweight_observer.add_energy()
        reweight_observer.enable_gradient()
        if cache_configuration:
            reweight_observer.cache_configuration()

    # Sampling method
    if sampling_method == "sweep":
        showln("using sweep sampling")
        sampling = SweepSampling(state, configuration_cut_dimension, restrict)
        sampling_total_step = sampling_total_step
        # Initialize sweep configuration
        if sweep_initial_configuration == "direct":
            # Use direct sampling to find sweep sampling initial configuration.
            direct_sampling = DirectSampling(state, configuration_cut_dimension, restrict,
                                             direct_sampling_cut_dimension)
            with seed_differ:
                _, configuration = direct_sampling()
        elif sweep_initial_configuration == "load":
            with open(configuration_dump_file, "rb") as file:
                configurations = pickle.load(file)
            if len(configurations) < mpi_size:
                with seed_differ:
                    choose = TAT.random.uniform_int(0, len(configurations) - 1)()
            else:
                choose = mpi_rank
            config = configurations[choose]
            configuration = sampling.configuration
            for l1 in range(state.L1):
                for l2 in range(state.L2):
                    for orbit, edge_point in config[l1][l2].items():
                        configuration[l1, l2, orbit] = edge_point
        else:
            with seed_differ:
                initial_configuration_module = importlib.import_module(sweep_initial_configuration)
                configuration = initial_configuration_module.initial_configuration(state, configuration_cut_dimension)
        sampling.configuration = configuration
    elif sampling_method == "ergodic":
        showln("using ergodic sampling")
        sampling = ErgodicSampling(state, configuration_cut_dimension, restrict)
        sampling_total_step = sampling.total_step
    elif sampling_method == "direct":
        showln("using direct sampling")
        sampling = DirectSampling(state, configuration_cut_dimension, restrict, direct_sampling_cut_dimension)
        sampling_total_step = sampling_total_step
    else:
        raise ValueError("Invalid sampling method")

    # Main loop
    with SignalHandler(signal.SIGINT) as sigint_handler:
        for grad_step in range(grad_total_step):
            if need_reweight_observer:
                configuration_pool = []
            # Sampling and observe
            log_prod_ws = 0.0
            with seed_differ, observer:
                for sampling_step in range(sampling_total_step):
                    if sampling_step % mpi_size == mpi_rank:
                        possibility, configuration = sampling()
                        log_prod_ws += np.log(np.abs(complex(configuration.hole(())).real))
                        observer(possibility, configuration)
                        if need_reweight_observer:
                            configuration_pool.append((possibility, configuration))
                        show(
                            f"sampling, total_step={sampling_total_step}, energy={observer.energy}, step={sampling_step}"
                        )
            log_prod_ws = mpi_comm.allreduce(log_prod_ws)
            showln(f"sampling done, total_step={sampling_total_step}, energy={observer.energy}")

            # Measure log
            if measurement and mpi_rank == 0:
                for measurement_name in measurement_names:
                    measurement_result = observer.result[measurement_name]
                    measurement_modules[measurement_name].save_result(state, measurement_result, grad_step)
            # Energy log
            if log_file and mpi_rank == 0:
                with open(log_file.replace("%s", str(grad_step)).replace("%t", time_str), "a",
                          encoding="utf-8") as file:
                    print(*observer.energy, file=file)
            # Dump configuration
            if configuration_dump_file:
                if sampling_method == "sweep":
                    to_dump = mpi_comm.gather(sampling.configuration._configuration, root=0)
                    if mpi_rank == 0:
                        with open(configuration_dump_file, "wb") as file:
                            pickle.dump(to_dump, file)
                else:
                    raise ValueError("Dump configuration into file is only supported for sweep sampling")

            if use_gradient:

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
                else:
                    if grad_step == 0 or momentum_parameter == 0.0:
                        total_grad = [[None for l2 in range(state.L2)] for l1 in range(state.L1)]
                        for l1 in range(state.L1):
                            for l2 in range(state.L2):
                                total_grad[l1][l2] = grad[l1][l2]
                    else:
                        if orthogonalize_momentum:
                            # _lattice_dot always return a real number
                            param = mpi_comm.bcast(observer._lattice_dot(total_grad, state._lattice) /
                                                   observer._lattice_dot(state._lattice, state._lattice),
                                                   root=0)
                            for l1 in range(state.L1):
                                for l2 in range(state.L2):
                                    total_grad[l1][l2] = total_grad[l1][l2] - state[l1, l2] * param
                        for l1 in range(state.L1):
                            for l2 in range(state.L2):
                                total_grad[l1][l2] = total_grad[l1][l2] * momentum_parameter + grad[l1][l2] * (
                                    1 - momentum_parameter)
                    if use_random_gradient:
                        showln("use random gradient")
                        this_grad = [[None for l2 in range(state.L2)] for l1 in range(state.L1)]
                        for l1 in range(state.L1):
                            for l2 in range(state.L2):
                                this_site_grad = total_grad[l1][l2]
                                random_same_shape = this_site_grad.same_shape().rand(0, 1)
                                random_same_shape.storage *= np.sign(this_site_grad.storage)
                                this_grad[l1][l2] = random_same_shape
                    else:
                        this_grad = total_grad
                    if use_fix_relative_step_size:
                        showln("fix relative step size")
                        param = mpi_comm.bcast((observer._lattice_dot(state._lattice, state._lattice) /
                                                observer._lattice_dot(this_grad, this_grad))**0.5,
                                               root=0)
                        real_step_size = grad_step_size * param
                    else:
                        real_step_size = grad_step_size
                    for l1 in range(state.L1):
                        for l2 in range(state.L2):
                            state[l1, l2] -= real_step_size * this_grad[l1][l2].conjugate(positive_contract=True)
                showln(f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}")

                # Fix gauge
                if fix_gauge:
                    fix_sampling_lattice_guage(state)
                # Normalize state
                try_normalize(state, log_prod_ws / sampling_total_step)
                # Bcast state and refresh sampling(refresh sampling aux and sampling config)
                bcast_lattice_buffer(state._lattice, root=0)
                sampling.refresh_all()

                # Save state
                if save_state_interval and (grad_step + 1) % save_state_interval == 0:
                    if save_state_file and mpi_rank == 0:
                        with open(save_state_file.replace("%s", str(grad_step)).replace("%t", time_str), "wb") as file:
                            pickle.dump(state, file)
            if sigint_handler():
                break
