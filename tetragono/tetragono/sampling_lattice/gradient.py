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

import signal
from datetime import datetime
import numpy as np
import TAT
from ..sampling_lattice import SamplingLattice, Observer, SweepSampling, ErgodicSampling, DirectSampling
from ..utility import (show, showln, mpi_rank, mpi_size, SignalHandler, seed_differ, lattice_randomize, write_to_file,
                       get_imported_function, restrict_wrapper, bcast_number, write_configurations)


def check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta):

    def get_energy():
        with energy_observer:
            for possibility, configuration in configuration_pool:
                configuration.refresh_all()
                energy_observer(possibility, configuration)
        energy, _ = energy_observer.total_energy
        return energy

    original_energy, _ = observer.total_energy
    delta = check_difference_delta
    showln(f"difference delta is set as {delta}")
    for l1, l2 in state.sites():
        showln(l1, l2)
        s = state[l1, l2].storage
        g = grad[l1][l2].transpose(state[l1, l2].names).storage
        for i in range(len(s)):
            value = s[i]
            s[i] = value + delta
            now_energy = get_energy()
            rgrad = (now_energy - original_energy) / delta
            if state.Tensor.is_complex:
                s[i] = value + delta * 1j
                now_energy = get_energy()
                igrad = (now_energy - original_energy) / delta
                cgrad = rgrad + igrad * 1j
            else:
                cgrad = rgrad
            s[i] = value
            showln(" ", abs(g[i] - cgrad) / abs(cgrad), cgrad, g[i])


def line_search(state, observer, grad, energy_observer, configuration_pool, step_size, line_search_amplitude):
    saved_state = [[state[l1, l2] for l2 in range(state.L2)] for l1 in range(state.L1)]

    def restore_state():
        for l1, l2 in state.sites():
            state[l1, l2] = saved_state[l1][l2]

    grad_dot_begin = bcast_number(state.lattice_dot(grad, observer.gradient))
    if grad_dot_begin > 0:
        for l1, l2 in state.sites():
            state[l1, l2] = state[l1, l2] - step_size * grad[l1][l2]
        with energy_observer:
            for possibility, configuration in configuration_pool:
                configuration.refresh_all()
                energy_observer(possibility, configuration)
                show(f"predicting eta={step_size}, energy={energy_observer.energy}")
        grad_dot_end = bcast_number(state.lattice_dot(grad, energy_observer.gradient))
        showln(f"predict eta={step_size}, energy={energy_observer.energy}, gradient dot={grad_dot_end}")
        restore_state()

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
        state: SamplingLattice,
        sampling_total_step=0,
        grad_total_step=1,
        grad_step_size=0,
        *,
        # About observer
        cache_configuration=False,
        classical_energy=None,
        # About sampling
        sampling_method="direct",
        configuration_cut_dimension=None,
        direct_sampling_cut_dimension=4,
        sampling_configurations=np.zeros(0, dtype=np.int64),
        sweep_hopping_hamiltonians=None,
        # About subspace
        restrict_subspace=None,
        # About gradient method
        use_check_difference=False,
        use_line_search=False,
        use_fix_relative_step_size=False,
        use_random_gradient=False,
        momentum_parameter=0.0,
        # About natural gradient
        use_natural_gradient=False,
        conjugate_gradient_method_step=20,
        conjugate_gradient_method_error=0.0,
        cache_natural_delta=None,
        use_natural_gradient_by_direct_pseudo_inverse=False,
        scalapack_libraries="libscalapack.so",
        natural_gradient_r_pinv=1e-12,
        natural_gradient_a_pinv=0,
        # About gauge fixing
        fix_gauge=False,
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
        check_difference_delta=1e-8,
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

    # About observer
    cache_configuration : bool | "drop", default=False
        Whether to cache the auxiliary tensors of configurations. If it is True, all auxiliary tensors would be saved
        during single gradient descent step. If it is "drop", all auxiliary tensors would be saved during single
        sampling observation. If it is False, program could not measure long range observables.
    classical_energy : Callable | str | None, default=None:
        A module name or function defines the classical energy shift. It accepts configuration and returns a float as
        energy shift.

    # About sampling
    sampling_method : "direct" | "sweep" | "ergodic", default="direct"
        The sampling method, which could be one of direct, sweep and ergodic.
    configuration_cut_dimension : int | None, default=None
        The dimension cut in two line to one line step. If it is left default None, programs will break if the program
        try to calculate two line to one line.
    direct_sampling_cut_dimension : int, default=4
        The dimension cut in estimating environment during direct sampling.
    sampling_configurations : object, default=zero_configuration
        The initial configuration used in sweep sampling methods. All sampling methods will save the last configuration
        into this sampling_configurations variable. If the function is invoked from gm_run(_g) interface, this parameter
        should be omitted.
    sweep_hopping_hamiltonians : Callable | str | None, default=None
        The module name or function that set the sweep hopping hamiltonians. It accepts state and return a dictionary
        representing hamiltonians as dict[tuple[tuple[int, int, int], ...], Tensor]. If it is left as None, the
        hamiltonians of the state itself will be used.

    # About subspace
    restrict_subspace : Callable | str | None, default=None
        The restrict subspace during sampling configurations. It causes low efficient except for sweep sampling. Please
        notice that for sweep sampling, subspace restricting chould be achieved by setting proper sweep hopping
        hamiltonians sometimes, so maybe there is no need to set this parameter.

    # About gradient method
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
    conjugate_gradient_method_step : int, default=20
        The max step in conjugate gradient method which calculates naturual gradient. If it is set to -1, there is no
        step limit.
    conjugate_gradient_method_error : float, default=0.0
        The error tolerance in conjugate gradient method which calculates natural gradient. CG will break if step limit
        reached or error tolerance achieved.
    cache_natural_delta : str | None, default=None
        Whether to cache delta used in natural gradient. If it is left default None, all delta will be saved in memory
        during single gradient descent step. If it is set to a string, it means file path, all delta will be saved to
        file first during sampling configurations, and read it back when calculating natural gradient.
    use_natural_gradient_by_direct_pseudo_inverse : bool, default=False
        Whether to use pseudo inverse directly instead of conjugate method when calculating natural gradient.
    scalapack_libraries : str, default="libscalapack.so"
        The scalapack library path used in pseudo inverse for calculating natural gradient.
    natural_gradient_r_pinv : float, default=1e-12
        The r_pinv used in pseudo inverse for calculating natural gradient.
    natural_gradient_a_pinv : float, default=0
        The a_pinv used in pseudo inverse for calculating natural gradient.

    # About gauge fixing
    fix_gauge : bool, default=False
        Whether to fix gauge of tensor network between every gradient descent steps.

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

    # Restrict subspace
    if restrict_subspace is not None:
        origin_restrict = get_imported_function(restrict_subspace, "restrict")
        restrict = restrict_wrapper(origin_restrict)
    else:
        restrict = None

    # Classical energy
    if classical_energy is not None:
        classical_energy = get_imported_function(classical_energy, "classical_energy")

    # Prepare observers
    observer = Observer(
        state,
        enable_energy=True,
        enable_gradient=use_gradient,
        enable_natural_gradient=use_natural_gradient,
        cache_natural_delta=cache_natural_delta,
        cache_configuration=cache_configuration,
        restrict_subspace=restrict,
        classical_energy=classical_energy,
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
            cache_configuration=cache_configuration,
            restrict_subspace=restrict,
            classical_energy=classical_energy,
        )

    # Main loop
    with SignalHandler(signal.SIGINT) as sigint_handler:
        for grad_step in range(grad_total_step):
            if need_energy_observer:
                configuration_pool = []
            # Sampling and observe
            with seed_differ, observer:
                # Sampling method
                if sampling_method == "sweep":
                    if sweep_hopping_hamiltonians is not None:
                        hopping_hamiltonians = get_imported_function(sweep_hopping_hamiltonians,
                                                                     "hopping_hamiltonians")(state)
                    else:
                        hopping_hamiltonians = None
                    sampling = SweepSampling(state, configuration_cut_dimension, restrict, hopping_hamiltonians)
                    sampling_total_step = sampling_total_step
                    # Initial sweep configuration
                    sampling.configuration.import_configuration(sampling_configurations)
                elif sampling_method == "ergodic":
                    sampling = ErgodicSampling(state, configuration_cut_dimension, restrict)
                    sampling_total_step = sampling.total_step
                elif sampling_method == "direct":
                    sampling = DirectSampling(state, configuration_cut_dimension, restrict,
                                              direct_sampling_cut_dimension)
                    sampling_total_step = sampling_total_step
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
                # Save configuration
                if mpi_rank < sampling_total_step and sampling_method != "ergodic":
                    new_configurations = configuration.export_configuration()
                    sampling_configurations.resize(new_configurations.shape, refcheck=False)
                    np.copyto(sampling_configurations, new_configurations)
            showln(f"sampling done, total_step={sampling_total_step}, energy={observer.energy}")
            if sampling_method == "direct":
                showln(f"direct sampling instability is {observer.instability}")

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
                    if use_natural_gradient_by_direct_pseudo_inverse:
                        grad = observer.natural_gradient_by_direct_pseudo_inverse(natural_gradient_r_pinv,
                                                                                  natural_gradient_a_pinv,
                                                                                  scalapack_libraries.split(","))
                    else:
                        grad = observer.natural_gradient_by_conjugate_gradient(conjugate_gradient_method_step,
                                                                               conjugate_gradient_method_error)
                else:
                    grad = observer.gradient

                # Change state
                if use_check_difference:
                    showln("checking difference")
                    check_difference(state, observer, grad, energy_observer, configuration_pool, check_difference_delta)

                elif use_line_search:
                    showln("line searching")
                    grad *= (state.lattice_dot() / state.lattice_dot(grad, grad))**0.5
                    grad_step_size = line_search(state, observer, grad, energy_observer, configuration_pool,
                                                 grad_step_size, line_search_amplitude)
                    state.apply_gradient(grad, grad_step_size * line_search_parameter)
                else:
                    if grad_step == 0 or momentum_parameter == 0.0:
                        total_grad = grad
                    else:
                        if orthogonalize_momentum:
                            param = state.lattice_dot(total_grad) / state.lattice_dot()
                            total_grad -= state._lattice * param
                        total_grad = total_grad * momentum_parameter + grad * (1 - momentum_parameter)
                    if use_random_gradient:
                        this_grad = lattice_randomize(total_grad)
                    else:
                        this_grad = total_grad
                    if use_fix_relative_step_size:
                        this_grad *= (state.lattice_dot() / state.lattice_dot(this_grad, this_grad))**0.5
                    state.apply_gradient(this_grad, grad_step_size)
                showln(f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}")

                # Fix gauge
                if fix_gauge:
                    state.expand_dimension(1.0, 0)
                # Normalize state
                observer.normalize_lattice()
                # Bcast state
                state.bcast_lattice()

            # Yield the measurement result
            yield (measurement_whole_result, measurement_result)

            # Save state
            if save_state_file:
                write_to_file(state, save_state_file.replace("%s", str(grad_step)).replace("%t", time_str))
            if save_configuration_file:
                write_configurations(sampling_configurations,
                                     save_configuration_file.replace("%s", str(grad_step)).replace("%t", time_str))

            if sigint_handler():
                break
