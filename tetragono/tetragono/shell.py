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

import os
import sys
import cmd
from io import StringIO
import numpy as np
import TAT
from .utility import (mpi_rank, mpi_size, mpi_comm, write_to_file, read_from_file, show, showln, seed_differ,
                      get_imported_function, allgather_array, restrict_wrapper, write_configurations,
                      read_configurations)
from . import conversion
from .exact_state import ExactState
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice, Configuration as gm_Configuration, SweepSampling
from .sampling_lattice.gradient import gradient_descent as gm_gradient_descent


class Config():

    __slots__ = ["args", "kwargs"]

    @staticmethod
    def _parse(i):
        if i == "True":
            return True
        if i == "False":
            return False
        try:
            return int(i)
        except ValueError:
            pass
        try:
            return float(i)
        except ValueError:
            pass
        return i

    def __init__(self, line):
        self.args = []
        self.kwargs = {}
        for arg in line.split():
            if "=" in arg:
                k, v = arg.split("=")
                self.kwargs[k] = self._parse(v)
            else:
                v = self._parse(arg)
                self.args.append(v)


class AutoCmdMeta(type):

    def _auto_func_generator(name, doc):

        def _auto_func(self, line):
            config = Config(line)
            getattr(self, name)(*config.args, **config.kwargs)

        _auto_func.__doc__ = doc
        return _auto_func

    def __new__(cls, name, bases, attrs):
        auto_attrs = {}
        for key, value in attrs.items():
            if hasattr(value, "_auto_cmd_meta_mark"):
                auto_attrs[f"do_{key}"] = cls._auto_func_generator(key, value.__doc__)

        return type.__new__(cls, name, bases, attrs | auto_attrs)


class AutoCmd(cmd.Cmd, metaclass=AutoCmdMeta):

    @staticmethod
    def decorator(function):
        function._auto_cmd_meta_mark = None
        return function


class TetragonoCommandApp(AutoCmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = "TET> "
        self.stored_line = ""
        self.license = """
Copyright (C) 2019-2024 USTC-TNS Group
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
        if mpi_rank == 0:
            self.intro = """Welcome to the Tetragono shell. Type help or ? to list commands.""" + self.license + f"\nRandom seed is set as {seed_differ.seed}"

        self.su = None
        self.ex = None
        self.gm = None
        self.gm_conf = np.zeros(0, dtype=np.int64)

    def precmd(self, line):
        line = line.split("#")[0].strip()
        if line.endswith("\\"):
            self.stored_line = self.stored_line + line[:-1].strip() + " "
            self.prompt = ".... "
            return ""
        line = self.stored_line + line
        self.stored_line = ""

        self.prompt = "TET> "
        return line

    def emptyline(self):
        pass

    def do_shell(self, line):
        """
        Run shell command.
        """
        if mpi_rank == 0:
            os.system(line)
        mpi_comm.Barrier()

    def do_EOF(self, line):
        """
        Exit tetra shell.
        """
        if mpi_rank == 0:
            print("EOF")
        return True

    def do_exit(self, line):
        """
        Exit tetra shell.
        """
        return True

    def do_quit(self, line):
        """
        Exit tetra shell.
        """
        return True

    @AutoCmd.decorator
    def seed(self, random_seed):
        """
        Set random seed.

        Parameters
        ----------
        random_seed : int
            The new random seed.
        """
        TAT.random.seed(random_seed)

    @staticmethod
    def ex_ap_create(lattice_type, model_name, *args, **kwargs):
        abstract_state = get_imported_function(model_name, "abstract_state")
        if len(args) == 1 and args[0] == "help":
            showln(abstract_state.__doc__.replace("\n", "\n    "))
            return None
        state = lattice_type(abstract_state(*args, **kwargs))
        return state

    @AutoCmd.decorator
    def numpy_hamiltonian(self, file, model, *args, **kwargs):
        """
        Export hamiltonian as a numpy array to a file.

        Parameters
        ----------
        file : str | None
            The file that the hamiltonian is exported to.
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        state = self.ex_ap_create(lambda x: x, model, *args, **kwargs)
        result = state.numpy_hamiltonian()
        if result is not None and file is not None:
            write_to_file(result, file)
        return result

    @AutoCmd.decorator
    def ex_create(self, *args, **kwargs):
        """
        Create a state used for exact update.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        state = self.ex_ap_create(ExactState, *args, **kwargs)
        if state is not None:
            self.ex = state

    @AutoCmd.decorator
    def ex_dump(self, name):
        """
        Dump the exact update lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        if self.ex is None:
            showln("ex is None")
        else:
            write_to_file(self.ex, name)

    @AutoCmd.decorator
    def ex_load(self, name):
        """
        Load the exact update lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.ex = read_from_file(name)

    @AutoCmd.decorator
    def ex_update(self, total_step):
        """
        Do exact update.

        Parameters
        ----------
        total_step : int
            The update total step to do.
        """
        self.ex.update(total_step)

    @AutoCmd.decorator
    def ex_energy(self):
        """
        Calculate exact energy.
        """
        showln("Exact state energy is", self.ex.observe_energy())

    @staticmethod
    def su_gm_create(lattice_type, model_name, *args, **kwargs):
        abstract_lattice = get_imported_function(model_name, "abstract_lattice")
        if len(args) == 1 and args[0] == "help":
            showln(abstract_lattice.__doc__.replace("\n", "\n    "))
            return None
        state = lattice_type(abstract_lattice(*args, **kwargs))

        # pre normalize the tensor
        for l1, l2 in state.sites():
            state[l1, l2] /= state[l1, l2].norm_max()
        return state

    @AutoCmd.decorator
    def su_create(self, *args, **kwargs):
        """
        Create a lattice used for simple update.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        state = self.su_gm_create(SimpleUpdateLattice, *args, **kwargs)
        if state is not None:
            self.su = state

    @AutoCmd.decorator
    def su_dump(self, name):
        """
        Dump the simple update lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        if self.su is None:
            showln("su is None")
        else:
            write_to_file(self.su, name)

    @AutoCmd.decorator
    def su_load(self, name):
        """
        Load the simple update lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.su = read_from_file(name)

    @AutoCmd.decorator
    def su_update(self, total_step, delta_tau, new_dimension):
        """
        Do simple update.

        Parameters
        ----------
        total_step : int
            The simple update total step to do.
        delta_tau : float
            The imaginary time, delta tau.
        new_dimension : int | float
            The new cut dimension used in simple update, or the amplitude of dimension expandance.
        """
        self.su.update(total_step, delta_tau, new_dimension)

    @AutoCmd.decorator
    def su_energy(self, cut_dimension):
        """
        Calculate simple update lattice with double layer auxiliaries.

        Parameters
        ----------
        cut_dimension : int
            The cut_dimension used in double layer auxiliaries.
        """
        self.su.initialize_auxiliaries(cut_dimension)
        showln("Simple update lattice energy is", self.su.observe_energy())

    @AutoCmd.decorator
    def su_to_ex(self):
        """
        Convert simple update lattice to exact state.
        """
        self.ex = conversion.simple_update_lattice_to_exact_state(self.su)

    @AutoCmd.decorator
    def su_to_gm(self):
        """
        Convert simple update lattice to sampling lattice.
        """
        self.gm = conversion.simple_update_lattice_to_sampling_lattice(self.su)

    @AutoCmd.decorator
    def gm_create(self, *args, **kwargs):
        """
        Create a lattice used for gradient method.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        state = self.su_gm_create(SamplingLattice, *args, **kwargs)
        if state is not None:
            self.gm = state

    @AutoCmd.decorator
    def gm_conf_create(self, module_name, *args, **kwargs):
        """
        Create configuration of sampling lattice.

        Parameters
        ----------
        module_name : str
            The module name to create initial configuration of sampling lattice.
        args, kwargs
            Arguments passed to module configuration creater function.
        """
        initial_configuration = get_imported_function(module_name, "initial_configuration")
        if len(args) == 1 and args[0] == "help":
            showln(initial_configuration.__doc__.replace("\n", "\n    "))
            return
        configuration = gm_Configuration(self.gm, -1)
        with seed_differ:
            # This configuration should never be used, so cut dimension is -1
            configuration = initial_configuration(configuration, *args, **kwargs)
        self.gm_conf = configuration.export_configuration()

    @AutoCmd.decorator
    def gm_dump(self, name):
        """
        Dump the sampling lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        if self.gm is None:
            showln("gm is None")
        else:
            write_to_file(self.gm, name)

    @AutoCmd.decorator
    def gm_conf_dump(self, name):
        """
        Dump the sampling lattice configuration into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        write_configurations(self.gm_conf, name)

    @AutoCmd.decorator
    def gm_load(self, name):
        """
        Load the sampling lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.gm = read_from_file(name)

    @AutoCmd.decorator
    def gm_conf_load(self, name):
        """
        Load the sampling lattice configuration from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.gm_conf = read_configurations(name)

    @AutoCmd.decorator
    def gm_conf_load_compat(self, name):
        """
        Load the sampling lattice configuration from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = read_from_file(name)
        size = len(config)
        if size < mpi_size:
            with seed_differ:
                choose = TAT.random.uniform_int(0, size - 1)()
        else:
            choose = mpi_rank
        self.gm_conf = config[choose]

    @AutoCmd.decorator
    def gm_run(self, *args, **kwargs):
        for result in self.gm_run_g(*args, **kwargs):
            pass
        return result

    def gm_run_g(self, *args, **kwargs):
        yield from gm_gradient_descent(self.gm, *args, **kwargs, sampling_configurations=self.gm_conf)

    gm_run.__doc__ = gm_run_g.__doc__ = gm_gradient_descent.__doc__

    @AutoCmd.decorator
    def gm_conf_eq(self, step, configuration_cut_dimension, sweep_hopping_hamiltonians=None, restrict_subspace=None):
        """
        Equilibium the configuration of sampling lattice.

        Parameters
        ----------
        step : int
            The step of sweep sampling for each process.
        configuration_cut_dimension : int
            The dimension cut in auxiliary tensor network.
        sweep_hopping_hamiltonians : str, optional
            The sweep hopping hamiltonians setter module name.
        restrict_subspace : str, optional
            The subspace restricter module name.
        """
        with seed_differ:
            state = self.gm
            sampling_configurations = self.gm_conf
            # Restrict subspace
            if restrict_subspace is not None:
                origin_restrict = get_imported_function(restrict_subspace, "restrict")
                restrict = restrict_wrapper(origin_restrict)
            else:
                restrict = None
            # Initialize sampling object
            if sweep_hopping_hamiltonians is not None:
                hopping_hamiltonians = get_imported_function(sweep_hopping_hamiltonians, "hopping_hamiltonians")(state)
            else:
                hopping_hamiltonians = None
            sampling = SweepSampling(state, configuration_cut_dimension, restrict, hopping_hamiltonians)
            # Initial sweep configuration
            sampling.configuration.import_configuration(sampling_configurations)
            # Equilibium
            for sampling_step in range(step):
                possibility, configuration = sampling()
                show(f"equilibium {sampling_step}/{step}")
            # Save configuration
            new_configurations = configuration.export_configuration()
            sampling_configurations.resize(new_configurations.shape, refcheck=False)
            np.copyto(sampling_configurations, new_configurations)
            showln(f"equilibium done, total_step={step}")

    @AutoCmd.decorator
    def gm_clear_symmetry(self):
        """
        Clear the symmetry of sampling lattice.
        """
        self.gm = self.gm.clear_symmetry()

    @AutoCmd.decorator
    def gm_hamiltonian(self, model, *args, **kwargs):
        """
        Replace the hamiltonian of the sampling lattice with another one.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        new_state = self.ex_ap_create(lambda x: x, model, *args, **kwargs)
        self.gm._hamiltonians = new_state._hamiltonians

    @AutoCmd.decorator
    def gm_expand(self, new_dimension, epsilon):
        """
        Expand dimension of sampling lattice.

        Parameters
        ----------
        new_dimension : int | float
            The new dimension, or the amplitude of dimension expandance.
        epsilon : float
            The relative error added into tensor.
        """
        self.gm.expand_dimension(new_dimension, epsilon)

    @AutoCmd.decorator
    def gm_to_ex(self):
        """
        Convert sampling lattice to exact state.
        """
        self.ex = conversion.sampling_lattice_to_exact_state(self.gm)

    @AutoCmd.decorator
    def gm_to_su(self):
        """
        Convert sampling lattice to simple update lattice.
        """
        self.su = conversion.sampling_lattice_to_simple_update_lattice(self.gm)


class TetragonoScriptApp(TetragonoCommandApp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rawinput = False
        self.prompt = ""
        if mpi_rank == 0:
            self.intro = """Welcome to the tetragono shell.""" + self.license + f"\nRandom seed is set as {seed_differ.seed}"

    def precmd(self, line):
        line = line.strip()
        line = super().precmd(line).strip()
        if line != "":
            showln("TET> ", line, sep="")
        self.prompt = ""
        return line


# Run
if __name__ == "__main__":
    help_message = """usage:
    shell.py
    shell.py [-h | -help | --help]
    shell.py <script_file>
    shell.py -- script"""
    if len(sys.argv) == 1:
        TetragonoCommandApp().cmdloop()
    elif len(sys.argv) == 2:
        script_file_name = sys.argv[1]
        if script_file_name in ["-h", "--help", "-help"]:
            showln(help_message)
        else:
            with open(script_file_name, "r", encoding="utf-8") as script_file:
                sys.path.append(os.path.dirname(os.path.abspath(sys.argv[1])))
                TetragonoScriptApp(stdin=script_file).cmdloop()
    elif sys.argv[1] == "--":
        commands = " ".join(sys.argv[2:]).replace(" - ", "\n")
        script_file = StringIO(commands)
        TetragonoScriptApp(stdin=script_file).cmdloop()
    else:
        showln("shell.py: Error: unrecognized command-line option")
        showln(help_message)
        sys.exit(1)
    mpi_comm.Barrier()
else:
    app = TetragonoCommandApp()

    shell = app.do_shell
    seed = app.seed
    numpy_hamiltonian = app.numpy_hamiltonian

    ex_create = app.ex_create
    ex_dump = app.ex_dump
    ex_load = app.ex_load
    ex_update = app.ex_update
    ex_energy = app.ex_energy

    su_create = app.su_create
    su_dump = app.su_dump
    su_load = app.su_load
    su_update = app.su_update
    su_energy = app.su_energy
    su_to_ex = app.su_to_ex
    su_to_gm = app.su_to_gm

    gm_create = app.gm_create
    gm_conf_create = app.gm_conf_create
    gm_dump = app.gm_dump
    gm_conf_dump = app.gm_conf_dump
    gm_load = app.gm_load
    gm_conf_load = app.gm_conf_load
    gm_conf_load_compat = app.gm_conf_load_compat
    gm_run = app.gm_run
    gm_run_g = app.gm_run_g
    gm_conf_eq = app.gm_conf_eq
    gm_clear_symmetry = app.gm_clear_symmetry
    gm_hamiltonian = app.gm_hamiltonian
    gm_expand = app.gm_expand
    gm_to_ex = app.gm_to_ex
    gm_to_su = app.gm_to_su
