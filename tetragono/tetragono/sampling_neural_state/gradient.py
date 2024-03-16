#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from datetime import datetime
import numpy as np
import torch
import TAT
from ..sampling_neural_state import SamplingNeuralState, Observer, SweepSampling, DirectSampling, ErgodicSampling
from ..utility import (show, showln, mpi_rank, mpi_size, seed_differ, write_to_file, get_imported_function,
                       bcast_number, bcast_buffer, write_configurations, allreduce_number)


def check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta):

    def get_energy():
        with energy_observer:
            for configurations, _, weights, multiplicities in configuration_pool:
                amplitudes = state(configurations)
                energy_observer(configurations, amplitudes, weights, multiplicities)
        energy, _ = energy_observer.total_energy
        return energy

    original_energy, _ = observer.total_energy
    delta = check_difference_delta
    showln(f"difference delta is set as {delta}")
    for i in range(len(grad)):
        basis = torch.zeros_like(grad)
        basis[i] = 1
        state.apply_gradient(basis, -delta)
        now_energy = get_energy()
        mgrad = (now_energy - original_energy) / delta
        state.apply_gradient(basis, +delta)
        showln((grad[i] - mgrad) / mgrad, mgrad, grad[i])


def line_search(state, observer, grad, energy_observer, configuration_pool, step_size, line_search_amplitude):
    grad_dot_begin = state.state_dot(grad, observer.gradient)
    bcast_buffer(grad_dot_begin)
    if grad_dot_begin > 0:
        state.apply_gradient(grad, +step_size)
        with energy_observer:
            for configurations, _, weights, multiplicies in configuration_pool:
                amplitudes = state(configurations)
                energy_observer(configurations, amplitudes, weights, multiplicies)
                show(f"predicting eta={step_size}, energy={energy_observer.energy}")
        grad_dot_end = state.state_dot(grad, energy_observer.gradient)
        bcast_buffer(grad_dot_end)
        showln(f"predict eta={step_size}, energy={energy_observer.energy}, gradient dot={grad_dot_end}")
        state.apply_gradient(grad, -step_size)

        if grad_dot_end > 0:
            step_size *= line_search_amplitude
            showln(f"step_size is chosen as {step_size}, since grad_dot(begin) > 0, grad_dot(end) > 0")
        else:
            step_size /= line_search_amplitude
            showln(f"step_size is chosen as {step_size}, since grad_dot(begin) > 0, grad_dot(end) <= 0")
    else:
        step_size = step_size
        showln(f"step_size is chosen as {step_size}, since grad_dot(begin) < 0")

    return bcast_number(step_size)


def gradient_descent(
        state: SamplingNeuralState,
        sampling_total_step=0,
        grad_total_step=1,
        grad_step_size=0,
        *,
        # About sampling
        expect_unique_sampling_step=None,
        sampling_method="sweep",
        sampling_configurations=None,
        sweep_hopping_hamiltonians=None,
        sweep_alpha=1.0,
        sampling_batch_size=None,
        observing_batch_size=None,
        # About gradient method
        outside_optimizer=False,
        use_check_difference=False,
        use_line_search=False,
        use_fix_relative_step_size=False,
        use_random_gradient=False,
        momentum_parameter=0.0,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=None,
        conjugate_gradient_method_error=None,
        conjugate_gradient_method_epsilon=None,
        # About log and save state
        log_file=None,
        save_state_file=None,
        save_configuration_file=None,
        # About line search
        line_search_amplitude=1.2,
        line_search_parameter=0.6,
        # About momentum
        orthogonalize_momentum=False,
        # About check difference
        check_difference_delta=1e-6,
        # About Measurement
        measurement=None):
    """
    Gradient method on sampling lattice.

    Parameters
    ----------
    state : SamplingLattice
        The sampling lattice to do gradient descent, if the function is invoked from gm_run(_g) interface, this
        parameter should be omitted.
    sampling_total_step : int, default=0
        The sampling total step at each gradient descent step, if the sampling method is set to ergodic method, this
        parameter will be ignored.
    grad_total_step : int, default=1
        The gradient descent step. If user pass 0 to this parameter, it will be set as 1 again.
    grad_step_size : float, default=0
        The gradient descent step size. it is the absolute step size of the gradient descent, if
        `use_fix_relative_step_size` is set to True, and is the relative step size if that is set to False.

    # About sampling
    expect_unique_sampling_step : int, optional
        The expect unique sampling step count.
    sampling_method : "sweep" | "direct" | "ergodic", default="sweep"
        The sampling method, which could be one of sweep, direct and ergodic.
    sampling_configurations : object, default=zero_configuration
        The initial configuration used in sweep sampling methods. All sampling methods will save the last configuration
        into this sampling_configurations variable. If the function is invoked from gm_run(_g) interface, this parameter
        should be omitted.
    sweep_hopping_hamiltonians : Callable | str | None, default=None
        The module name or function that set the sweep hopping hamiltonians. It accepts state and return a dictionary
        representing hamiltonians as dict[tuple[tuple[int, int, int], ...], Tensor]. If it is left as None, the
        hamiltonians of the state itself will be used.
    sweep_alpha : float, default=1.0
        The alpha parameter in sweep sampling method.
    sampling_batch_size : int, optional
        The sampling batch size for single sampling.
    observing_batch_size : int, optional
        The observe batch size for single observing.

    # About gradient method
    outside_optimizer : bool, default=False
        Whether to use the pytorch optimizer instead of ours, this option set the gradient of the network parameters,
        and users need to update network manually by the pytorch optimizer.
    use_check_difference : bool, default=False
        Check the gradient with numeric difference. WARNING: do not enable it unless you know what you are doing, this
        option is used for internal debugging.
    use_line_search : bool, default=False
        Whether to deploy line search, if it is enabled, `use_fix_relative_step_size` will be set to True forcely,
        `momentum_parameter` and `use_random_gradient` will be ignored.
    use_fix_relative_step_size : bool, default=False
        Whether to use relative step size or absoluate step size in gradient descent. This option will be set to True
        forcely if line search enabled.
    use_random_gradient : bool, default=False
        Whether to use only the sign of the gradient but replace the absolute value as random numbers in uniform
        distribution between 0 and 1. After the absolute value reconstructed, relative or absoluate step size will be
        multipled on it. This parameter will be ignored if line search enabled.
    momentum_parameter : float, default=0.0
        Whether to deply momentum between gradient descent steps. The real update x will be `x' * p + g * (1-p)` where p
        is this `momentum_parameter`, x' is the real update in the last step, and g is the gradient. This parameter will
        be ignored if line search enabled.

    # About natural gradient
    use_natural_gradient : bool, default=False
        Whether to enable natural gradient instead of trivial gradient.
    conjugate_gradient_method_step : int, optional
        The max step in conjugate gradient method which calculates naturual gradient. If it is set to -1, there is no
        step limit.
    conjugate_gradient_method_error : float, optional
        The error tolerance in conjugate gradient method which calculates natural gradient. CG will break if step limit
        reached or error tolerance achieved.
    conjugate_gradient_method_epsilon : float, optional
        The epsilon to be added to the diagonal of the matrix during conjugate gradient method to avoid instability
    caused by small singular value.

    # About log and save state
    log_file : str, default=None
        The energy log file path during gradient descent.
    save_state_file : str, default=None
        The state data file path during gradient descent.
    save_configuration_file : str, default=None
        The configuration data file path during gradient descent.

    # About line search
    line_search_amplitude : float, default=1.2
        The amplitude used in line search
    line_search_parameter : float, default=0.6
        The parameter used in line search

    # About momentum
    orthogonalize_momentum : bool, default=False
        Whether to orthogonalize the momentum before calculating the real update.

    # About check difference
    check_difference_delta : float, default=1e-8
        The parameter used in check difference.

    # About Measurement
    measurement : str | Callable | list[str | Callable] | None, default=None
        The list or one of the module name or function that defines the observables which should be measured during
        sampling configurations. If it is module, the function `save_result` in the same module will be called with
        measurement result after measured. If it is callable directly, the only way to retrieve the result is to get the
        yield result of this function. To get the yield result, consider using `gm_run_g` instead if you are using
        `gm_run`.
    """

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

    # Prepare observers
    observer = Observer(
        state,
        enable_energy=True,
        enable_gradient=use_gradient,
        enable_natural_gradient=use_natural_gradient,
    )
    if measurement:
        if isinstance(measurement, str):
            # It is measurement modules names joined by ","
            measurement = measurement.split(",")
        if not isinstance(measurement, list):
            measurement = [measurement]
        # It is a python list of measurement modules names or function directly.
        for measure_term in measurement:
            if isinstance(measure_term, str):
                observer.add_observer(measure_term, get_imported_function(measure_term, "measurement")(state))
            else:
                observer.add_observer(measure_term.__name__, measure_term(state))
    if use_gradient:
        need_energy_observer = use_line_search or use_check_difference
    else:
        need_energy_observer = False
    if need_energy_observer:
        energy_observer = Observer(
            state,
            enable_energy=True,
            enable_gradient=use_line_search,
        )

    # Main loop
    for grad_step in range(grad_total_step):
        if need_energy_observer:
            configuration_pool = []
        # Sampling and observe
        with seed_differ, observer:
            torch.manual_seed(seed_differ.random_int())
            # Sampling method
            if sampling_method == "sweep":
                if sweep_hopping_hamiltonians is not None:
                    hopping_hamiltonians = get_imported_function(sweep_hopping_hamiltonians,
                                                                 "hopping_hamiltonians")(state)
                else:
                    hopping_hamiltonians = None
                sampling = SweepSampling(
                    state,
                    sampling_configurations,
                    sampling_total_step,
                    hopping_hamiltonians,
                    sweep_alpha,
                )
                configurations_pool, amplitudes_pool, weights_pool, multiplicities_pool = sampling()
            elif sampling_method == "direct":
                sampling = DirectSampling(state, sampling_total_step, sweep_alpha)
                configurations_pool, amplitudes_pool, weights_pool, multiplicities_pool = sampling()
            elif sampling_method == "ergodic":
                sampling = ErgodicSampling(state)
                configurations_pool, amplitudes_pool, weights_pool, multiplicities_pool = sampling()
            else:
                raise ValueError("Invalid sampling method")
            # Save configuration
            new_configurations = configurations_pool.cpu().numpy()
            sampling_configurations.resize(new_configurations.shape, refcheck=False)
            np.copyto(sampling_configurations, new_configurations)
            # Observe
            unique_sampling_count = len(multiplicities_pool)
            total_unique_sampling_count = int(allreduce_number(unique_sampling_count))
            showln(f"sampling done, unique {total_unique_sampling_count}")
            if observing_batch_size is None:
                batch_size = unique_sampling_count
            else:
                batch_size = observing_batch_size
            for unique_sampling_step in range(0, unique_sampling_count, batch_size):
                process = unique_sampling_step / unique_sampling_count
                show(f"observing {100*process:.2f}%, energy={observer.energy}")
                configurations = configurations_pool[unique_sampling_step:unique_sampling_step + batch_size]
                amplitudes = amplitudes_pool[unique_sampling_step:unique_sampling_step + batch_size]
                weights = weights_pool[unique_sampling_step:unique_sampling_step + batch_size]
                multiplicities = multiplicities_pool[unique_sampling_step:unique_sampling_step + batch_size]
                observer(configurations, amplitudes, weights, multiplicities)
                if need_energy_observer:
                    configuration_pool.append((configurations, amplitudes, weights, multiplicities))
        torch.manual_seed(seed_differ.random_int())
        showln(f"observing done, total_step={observer.count}, energy={observer.energy}")
        if expect_unique_sampling_step is not None:
            sampling_total_step = int(sampling_total_step * expect_unique_sampling_step / total_unique_sampling_count)
            showln(f"sampling total step update to {sampling_total_step}")

        # Measure log
        measurement_result = observer.result
        measurement_whole_result = observer.whole_result
        if measurement is not None and mpi_rank == 0:
            for measure_term in measurement:
                # If measure_term is not a module name but a function directly,
                # it is only used when setting the measurement.
                if isinstance(measure_term, str):
                    save_result = get_imported_function(measure_term, "save_result")
                    save_result(
                        state,
                        measurement_result[measure_term],
                        measurement_whole_result[measure_term],
                    )
        # Energy log
        if log_file and mpi_rank == 0:
            with open(log_file.replace("%t", time_str), "a", encoding="utf-8") as file:
                print(*observer.energy, file=file)

        if use_gradient:

            # Get gradient
            if use_natural_gradient:
                grad = observer.natural_gradient(
                    conjugate_gradient_method_step,
                    conjugate_gradient_method_error,
                    conjugate_gradient_method_epsilon,
                )
            else:
                grad = observer.gradient

            # Change state
            if outside_optimizer:
                state.set_gradient(grad)
            elif use_check_difference:
                showln("checking difference")
                check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta)

            elif use_line_search:
                showln("line searching")
                grad *= (state.state_dot() / state.state_dot(grad, grad))**0.5
                grad_step_size = line_search(state, observer, grad, energy_observer, configuration_pool, grad_step_size,
                                             line_search_amplitude)
                state.apply_gradient(grad, grad_step_size * line_search_parameter)
            else:
                if grad_step == 0 or momentum_parameter == 0.0:
                    total_grad = grad
                else:
                    if orthogonalize_momentum:
                        param = state.state_dot(total_grad) / state.state_dot()
                        total_grad -= state.state_vector() * param
                    total_grad = total_grad * momentum_parameter + grad * (1 - momentum_parameter)
                if use_random_gradient:
                    this_grad = total_grad.sgn() * torch.rand_like(total_grad)
                else:
                    this_grad = total_grad
                if use_fix_relative_step_size:
                    this_grad *= (state.state_dot() / state.state_dot(this_grad, this_grad))**0.5
                state.apply_gradient(this_grad, grad_step_size)
            showln(f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}")

            # Bcast state
            state.bcast_state()

        # Yield the measurement result
        yield (measurement_whole_result, measurement_result)

        # Save state
        if save_state_file:
            torch.save(state.network.state_dict(),
                       save_state_file.replace("%s", str(grad_step)).replace("%t", time_str))
        if save_configuration_file:
            write_configurations(sampling_configurations,
                                 save_configuration_file.replace("%s", str(grad_step)).replace("%t", time_str))
