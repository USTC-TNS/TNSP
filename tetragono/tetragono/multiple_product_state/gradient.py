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
from ..multiple_product_state import MultipleProductState, Sampling, Observer
from ..common_toolkit import SignalHandler, seed_differ, mpi_comm, mpi_size, mpi_rank, show, showln, write_to_file, get_imported_function


def gradient_descent(
        state: MultipleProductState,
        sampling_total_step,
        grad_total_step,
        grad_step_size,
        *,
        # About sampling
        sampling_configurations=[],
        sweep_hopping_hamiltonians=None,
        restrict_subspace=None,
        # About gradient
        enable_gradient_ansatz=None,
        use_fix_relative_step_size=False,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=20,
        metric_inverse_epsilon=0.01,
        cache_natural_delta=None,
        # About log and save state
        log_file=None,
        save_state_interval=None,
        save_state_file=None,
        # About Measurement
        measurement=None):

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
    if grad_step_size != 0:
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

    # Gradient descent
    with SignalHandler(signal.SIGINT) as sigint_handler:
        for grad_step in range(grad_total_step):
            with observer, seed_differ:
                # Create sampling object
                if sweep_hopping_hamiltonians is not None:
                    hopping_hamiltonians = get_imported_function(sweep_hopping_hamiltonians,
                                                                 "hopping_hamiltonians")(state)
                else:
                    hopping_hamiltonians = None
                if len(sampling_configurations) < mpi_size:
                    choose = TAT.random.uniform_int(0, len(sampling_configurations) - 1)()
                else:
                    choose = mpi_rank
                sampling = Sampling(state,
                                    configuration=sampling_configurations[choose],
                                    hopping_hamiltonians=hopping_hamiltonians,
                                    restrict_subspace=restrict)
                # Sampling
                for sampling_step in range(sampling_total_step):
                    if sampling_step % mpi_size == mpi_rank:
                        observer(sampling.configuration)
                        for _ in range(state.site_number):
                            sampling()
                        show(f"sampling {sampling_step}/{sampling_total_step}, energy={observer.energy}")
                # Save configurations
                gathered_configurations = mpi_comm.allgather(sampling.configuration)
                sampling_configurations.clear()
                sampling_configurations += gathered_configurations
            showln(f"gradient {grad_step}/{grad_total_step}, energy={observer.energy}")
            # Measure log
            if measurement and mpi_rank == 0:
                for measurement_name in measurement_names:
                    measurement_result = observer.result[measurement_name]
                    get_imported_function(measurement_name, "save_result")(state, measurement_result, grad_step)
            # Energy log
            if log_file and mpi_rank == 0:
                with open(log_file, "a", encoding="utf-8") as file:
                    print(*observer.energy, file=file)
            # Update state
            if use_natural_gradient:
                gradient = observer.natural_gradient(conjugate_gradient_method_step, metric_inverse_epsilon)
                showln("calculate natural gradient done")
            else:
                gradient = observer.gradient
            state.apply_gradient(gradient, grad_step_size, relative=use_fix_relative_step_size)
            # Save state
            if save_state_interval and (grad_step + 1) % save_state_interval == 0 and save_state_file:
                write_to_file(state, save_state_file)

            if sigint_handler():
                break
