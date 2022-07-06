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

import signal
import inspect
import TAT
from ..multiple_product_state import MultipleProductState, MetropolisSampling, ErgodicSampling, Observer
from ..common_toolkit import SignalHandler, seed_differ, mpi_comm, mpi_size, mpi_rank, show, showln, write_to_file, get_imported_function


def check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta):

    def get_energy():
        state.refresh_auxiliaries()
        with energy_observer:
            for possibility, configuration in configuration_pool:
                energy_observer(possibility, configuration)
        return energy_observer.energy[0] * state.site_number

    original_energy = observer.energy[0] * state.site_number
    delta = check_difference_delta
    showln(f"difference delta is set as {delta}")
    for name in state.ansatzes:
        showln(name)
        data = state.ansatzes[name].export_data()
        for i, tensor in enumerate(data):
            showln(f" {i}")
            if hasattr(tensor, "storage"):
                s = tensor.storage
                g = grad[name][i].transpose(tensor.names).storage
            else:
                s = tensor.reshape([-1])
                g = grad[name][i].reshape([-1])
            for i in range(len(s)):
                s[i] += delta
                state.ansatzes[name].import_data(data)
                now_energy = get_energy()
                rgrad = (now_energy - original_energy) / delta
                s[i] -= delta
                state.ansatzes[name].import_data(data)
                if state.Tensor.is_complex:
                    s[i] += delta * 1j
                    state.ansatzes[name].import_data(data)
                    now_energy = get_energy()
                    igrad = (now_energy - original_energy) / delta
                    s[i] -= delta * 1j
                    state.ansatzes[name].import_data(data)
                    cgrad = rgrad + igrad * 1j
                else:
                    cgrad = rgrad
                showln(" ", g[i] / cgrad, cgrad, g[i])


def line_search(state, observer, grad, energy_observer, configuration_pool, step_size, line_search_amplitude,
                line_search_error_threshold):
    saved_state = {name: state.ansatzes[name].export_data().copy() for name in state.ansatzes}
    grad_dot_pool = {}

    def grad_dot(eta):
        if eta not in grad_dot_pool:
            for name in state.ansatzes:
                state.ansatzes[name].import_data(saved_state[name] - eta * grad[name])
            with energy_observer:
                for possibility, configuration in configuration_pool:
                    energy_observer(possibility, configuration)
                    show(f"predicting eta={eta}, energy={energy_observer.energy}")
            result = mpi_comm.bcast(observer.delta_dot_sum(grad, energy_observer.gradient))
            showln(f"predict eta={eta}, energy={energy_observer.energy}, gradient dot={result}")
            grad_dot_pool[eta] = result
        return grad_dot_pool[eta]

    grad_dot_pool[0] = mpi_comm.bcast(observer.delta_dot_sum(grad, observer.gradient))
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

    for name in state.ansatzes:
        state.ansatzes[name].import_data(saved_state[name])
    return mpi_comm.bcast(step_size)


def gradient_descent(
        state: MultipleProductState,
        sampling_total_step,
        grad_total_step,
        grad_step_size,
        *,
        # About sampling
        sampling_method="metropolis",
        sampling_configurations=[],
        sweep_hopping_hamiltonians=None,
        metropolis_interval=1,
        # About subspace
        restrict_subspace=None,
        # About gradient method
        enable_gradient_ansatz=None,
        use_check_difference=False,
        use_line_search=False,
        use_fix_relative_step_size=False,
        momentum_parameter=0.0,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=20,
        metric_inverse_epsilon=0.01,
        cache_natural_delta=None,
        # About log and save state
        log_file=None,
        save_state_interval=None,
        save_state_file=None,
        # About line search
        line_search_amplitude=1.25,
        line_search_error_threshold=0.1,
        # About check difference
        check_difference_delta=1e-8,
        # About Measurement
        measurement=None):

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

    # Restrict subspace
    if restrict_subspace is not None:
        origin_restrict = get_imported_function(restrict_subspace, "restrict")
        if len(inspect.signature(origin_restrict).parameters) == 1:

            def restrict(configuration, replacement=None):
                if replacement is None:
                    return origin_restrict(configuration)
                else:
                    configuration = configuration.copy()
                    for [l1, l2, orbit], new_site_config in replacement.items():
                        configuration[l1, l2, orbit] = new_site_config
                    return origin_restrict(configuration)
        else:
            restrict = origin_restrict
    else:
        restrict = None

    # Create observer
    observer = Observer(state)
    observer.restrict_subspace(restrict)
    observer.add_energy()
    if use_gradient:
        if enable_gradient_ansatz is not None:
            observer.enable_gradient(enable_gradient_ansatz.split(","))
        else:
            observer.enable_gradient()
        if use_natural_gradient:
            observer.enable_natural_gradient()
            observer.cache_natural_delta(cache_natural_delta)
    if measurement:
        measurement_names = measurement.split(",")
        for measurement_name in measurement_names:
            observer.add_observer(measurement_name, get_imported_function(measurement_name, "measurement")(state))
    if use_gradient:
        need_energy_observer = use_line_search or use_check_difference
    else:
        need_energy_observer = False
    if need_energy_observer:
        energy_observer = Observer(state)
        energy_observer.restrict_subspace(restrict)
        energy_observer.add_energy()
        if use_line_search:
            if enable_gradient_ansatz is not None:
                energy_observer.enable_gradient(enable_gradient_ansatz.split(","))
            else:
                energy_observer.enable_gradient()

    # Main loop
    with SignalHandler(signal.SIGINT) as sigint_handler:
        for grad_step in range(grad_total_step):
            if need_energy_observer:
                configuration_pool = []
            # Sampling and observe
            with observer, seed_differ:
                # Sampling method
                if sampling_method == "metropolis":
                    if sweep_hopping_hamiltonians is not None:
                        hopping_hamiltonians = get_imported_function(sweep_hopping_hamiltonians,
                                                                     "hopping_hamiltonians")(state)
                    else:
                        hopping_hamiltonians = None
                    if len(sampling_configurations) < mpi_size:
                        choose = TAT.random.uniform_int(0, len(sampling_configurations) - 1)()
                    else:
                        choose = mpi_rank
                    sampling = MetropolisSampling(state,
                                                  restrict_subspace=restrict,
                                                  configuration=sampling_configurations[choose],
                                                  hopping_hamiltonians=hopping_hamiltonians,
                                                  interval=metropolis_interval)
                elif sampling_method == "ergodic":
                    sampling = ErgodicSampling(state, restrict_subspace=restrict)
                    sampling_total_step = sampling.total_step
                else:
                    raise ValueError("Invalid sampling method")
                # Sampling run
                for sampling_step in range(sampling_total_step):
                    if sampling_step % mpi_size == mpi_rank:
                        possibility, configuration = sampling()
                        observer(possibility, configuration)
                        if need_energy_observer:
                            configuration_pool.append((possibility, configuration))
                        show(f"sampling {sampling_step}/{sampling_total_step}, energy={observer.energy}")
                # Save configurations
                gathered_configurations = mpi_comm.allgather(sampling.configuration)
                sampling_configurations.clear()
                sampling_configurations += gathered_configurations
            showln(f"sampling done, total_step={sampling_total_step}, energy={observer.energy}")

            # Measure log
            if measurement and mpi_rank == 0:
                for measurement_name in measurement_names:
                    measurement_result = observer.result[measurement_name]
                    get_imported_function(measurement_name, "save_result")(state, measurement_result, grad_step)
            # Energy log
            if log_file and mpi_rank == 0:
                with open(log_file, "a", encoding="utf-8") as file:
                    print(*observer.energy, file=file)

            if use_gradient:
                # Get gradient
                if use_natural_gradient:
                    show("calculating natural gradient")
                    grad = observer.natural_gradient(conjugate_gradient_method_step, metric_inverse_epsilon)
                    showln("calculate natural gradient done")
                else:
                    grad = observer.gradient

                # Change state
                if use_check_difference:
                    showln("checking difference")
                    check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta)

                elif use_line_search:
                    showln("line searching")
                    grad = state.fix_relative_to_state(grad)
                    grad_step_size = line_search(state, observer, grad, energy_observer, configuration_pool,
                                                 grad_step_size, line_search_amplitude, line_search_error_threshold)
                    state.apply_gradient(grad, grad_step_size)
                else:
                    if grad_step == 0 or momentum_parameter == 0.0:
                        total_grad = grad
                    else:
                        total_grad = total_grad * momentum_parameter + grad * (1 - momentum_parameter)
                    this_grad = total_grad
                    if use_fix_relative_step_size:
                        this_grad = state.fix_relative_to_state(this_grad)
                    state.apply_gradient(grad, grad_step_size)
                showln(f"gradient {grad_step}/{grad_total_step}, step_size={grad_step_size}")

                # Save state
                if save_state_interval and (grad_step + 1) % save_state_interval == 0 and save_state_file:
                    write_to_file(state, save_state_file)

            if sigint_handler():
                break
