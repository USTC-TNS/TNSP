#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import TAT
from io import StringIO
from .common_toolkit import (mpi_rank, mpi_size, mpi_comm, write_to_file, read_from_file, show, showln, seed_differ,
                             get_imported_function, seed_differ)
from . import conversion
from .exact_state import ExactState
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice, Configuration as gm_Configuration
from .ansatz_product_state import AnsatzProductState, Configuration as ap_Configuration
from .sampling_lattice.gradient import gradient_descent as gm_gradient_descent
from .ansatz_product_state.gradient import gradient_descent as ap_gradient_descent


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
Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
        if mpi_rank == 0:
            self.intro = """Welcome to the Tetragono shell. Type help or ? to list commands.""" + self.license + f"\nRandom seed is set as {seed_differ.seed}"

        self.su = None
        self.ex = None
        self.gm = None
        self.gm_conf = []
        self.ap = None
        self.ap_conf = []

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
        mpi_comm.barrier()

    def do_EOF(self, line):
        """
        Exit tetra shell.
        """
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
        else:
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

    @staticmethod
    def su_gm_create(lattice_type, model_name, *args, **kwargs):
        abstract_lattice = get_imported_function(model_name, "abstract_lattice")
        if len(args) == 1 and args[0] == "help":
            showln(abstract_lattice.__doc__.replace("\n", "\n    "))
            return None
        else:
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
        new_dimension : int
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
        Convert simple update lattice to exact lattice.
        """
        self.ex = conversion.simple_update_lattice_to_exact_state(self.su)

    @AutoCmd.decorator
    def su_to_gm(self):
        """
        Convert simple update lattice to sampling lattice.
        """
        self.gm = conversion.simple_update_lattice_to_sampling_lattice(self.su)

    @AutoCmd.decorator
    def ex_update(self, total_step, approximate_energy):
        """
        Do exact update.

        Parameters
        ----------
        total_step : int
            The update total step to do.
        approximate_energy : float
            The approximate energy per site, it should ensure the ground state energy is the largest after shifting.
        """
        self.ex.update(total_step, approximate_energy)

    @AutoCmd.decorator
    def ex_energy(self):
        """
        Calculate exact energy.
        """
        showln("Exact state energy is", self.ex.observe_energy())

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
    def gm_run(self, *args, **kwargs):
        """
        Do gradient descent. see sampling_lattice/gradient.py for details.
        """
        for _ in gm_gradient_descent(self.gm, *args, **kwargs, sampling_configurations=self.gm_conf):
            pass

    def gm_run_g(self, *args, **kwargs):
        yield from gm_gradient_descent(self.gm, *args, **kwargs, sampling_configurations=self.gm_conf)

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
        if self.gm_conf is None:
            showln("gm_conf is None")
        else:
            write_to_file(self.gm_conf, name)

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
        self.gm_conf = read_from_file(name)

    @AutoCmd.decorator
    def gm_conf_create(self, module_name):
        """
        Create configuration of sampling lattice.

        Parameters
        ----------
        module_name : str
            The module name to create initial configuration of sampling lattice.
        """
        with seed_differ:
            # This configuration should never be used, so cut dimension is -1
            configuration = gm_Configuration(self.gm, -1)
            configuration = get_imported_function(module_name, "initial_configuration")(configuration)
            self.gm_conf = mpi_comm.allgather(configuration.export_configuration())

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
        Convert sampling lattice to exact lattice.
        """
        self.ex = conversion.sampling_lattice_to_exact_state(self.gm)

    @AutoCmd.decorator
    def ap_create(self, *args, **kwargs):
        """
        Create a ansatz product state.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        state = self.ex_ap_create(AnsatzProductState, *args, **kwargs)
        if state is not None:
            self.ap = state

    @AutoCmd.decorator
    def ap_dump(self, name):
        """
        Dump the ansatz product state into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        if self.ap is None:
            showln("ap is None")
        else:
            write_to_file(self.ap, name)

    @AutoCmd.decorator
    def ap_load(self, name):
        """
        Load the ansatz product state from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.ap = read_from_file(name)

    @AutoCmd.decorator
    def ap_ansatz_set(self, name, ansatz, *args, **kwargs):
        """
        Set the ansatz for ansatz product state.

        Parameters
        ----------
        name : str
            The subansatz name in this state.
        ansatz : str
            The ansatz names.
        args, kwargs
            Arguments passed to ansatz creater function.
        """
        create_ansatz = get_imported_function(ansatz, "ansatz")
        if len(args) == 1 and args[0] == "help":
            showln(create_ansatz.__doc__.replace("\n", "\n    "))
        else:
            self.ap.set_ansatz(create_ansatz(self.ap, *args, **kwargs), name)

    @AutoCmd.decorator
    def ap_ansatz_add(self, name, ansatz, *args, **kwargs):
        """
        Add the ansatz for ansatz product state.

        Parameters
        ----------
        name : str
            The subansatz name in this state.
        ansatz : str
            The ansatz names.
        args, kwargs
            Arguments passed to ansatz creater function.
        """
        create_ansatz = get_imported_function(ansatz, "ansatz")
        if len(args) == 1 and args[0] == "help":
            showln(create_ansatz.__doc__.replace("\n", "\n    "))
        else:
            self.ap.add_ansatz(create_ansatz(self.ap, *args, **kwargs), name)

    @AutoCmd.decorator
    def ap_ansatz_mul(self, name, ansatz, *args, **kwargs):
        """
        Mul the ansatz for ansatz product state.

        Parameters
        ----------
        name : str
            The subansatz name in this state.
        ansatz : str
            The ansatz names.
        args, kwargs
            Arguments passed to ansatz creater function.
        """
        create_ansatz = get_imported_function(ansatz, "ansatz")
        if len(args) == 1 and args[0] == "help":
            showln(create_ansatz.__doc__.replace("\n", "\n    "))
        else:
            self.ap.mul_ansatz(create_ansatz(self.ap, *args, **kwargs), name)

    @AutoCmd.decorator
    def ap_ansatz_show(self):
        """
        Show the ansatz for ansatz product state.
        """
        self.ap.show_ansatz()

    @AutoCmd.decorator
    def ap_ansatz_lock(self, path=""):
        """
        Lock the ansatz for ansatz product state.

        Parameters
        ----------
        path : str, default=""
            The path of ansatz to lock.
        """
        self.ap.ansatz.lock(path)

    @AutoCmd.decorator
    def ap_ansatz_unlock(self, path=""):
        """
        Unlock the ansatz for ansatz product state.

        Parameters
        ----------
        path : str, default=""
            The path of ansatz to unlock.
        """
        self.ap.ansatz.unlock(path)

    @AutoCmd.decorator
    def ap_run(self, *args, **kwargs):
        """
        Do gradient descent on ansatz product state. see ansatz_product_state/gradient.py for details.
        """
        for _ in ap_gradient_descent(self.ap, *args, **kwargs, sampling_configurations=self.ap_conf):
            pass

    def ap_run_g(self, *args, **kwargs):
        yield from ap_gradient_descent(self.ap, *args, **kwargs, sampling_configurations=self.ap_conf)

    @AutoCmd.decorator
    def ap_conf_create(self, module_name):
        """
        Create configuration of ansatz product state.

        Parameters
        ----------
        module_name : str
            The module name to create initial configuration of ansatz product state.
        """
        with seed_differ:
            configuration = ap_Configuration(self.ap)
            configuration = get_imported_function(module_name, "initial_configuration")(configuration)
            self.ap_conf = mpi_comm.allgather(configuration.export_configuration())

    @AutoCmd.decorator
    def ap_conf_dump(self, name):
        """
        Dump the ansatz product state configuration into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        if self.ap_conf is None:
            showln("ap_conf is None")
        else:
            write_to_file(self.ap_conf, name)

    @AutoCmd.decorator
    def ap_conf_load(self, name):
        """
        Load the ansatz product state configuration from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        self.ap_conf = read_from_file(name)

    @AutoCmd.decorator
    def ap_hamiltonian(self, model, *args, **kwargs):
        """
        Replace the hamiltonian of the ansatz product state with another one.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        new_state = self.ex_ap_create(lambda x: x, model, *args, **kwargs)
        self.ap._hamiltonians = new_state._hamiltonians


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
    shell.py script_file
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
    mpi_comm.barrier()
else:
    app = TetragonoCommandApp()

    seed = app.seed
    shell = app.do_shell
    numpy_hamiltonian = app.numpy_hamiltonian

    su_create = app.su_create
    su_dump = app.su_dump
    su_load = app.su_load
    su_update = app.su_update
    su_energy = app.su_energy
    su_to_ex = app.su_to_ex
    su_to_gm = app.su_to_gm

    ex_create = app.ex_create
    ex_update = app.ex_update
    ex_energy = app.ex_energy
    ex_dump = app.ex_dump
    ex_load = app.ex_load

    gm_create = app.gm_create
    gm_run = app.gm_run
    gm_dump = app.gm_dump
    gm_load = app.gm_load
    gm_expand = app.gm_expand
    gm_to_ex = app.gm_to_ex
    gm_conf_dump = app.gm_conf_dump
    gm_conf_load = app.gm_conf_load
    gm_conf_create = app.gm_conf_create
    gm_hamiltonian = app.gm_hamiltonian
    gm_clear_symmetry = app.gm_clear_symmetry

    ap_create = app.ap_create
    ap_dump = app.ap_dump
    ap_load = app.ap_load
    ap_ansatz_set = app.ap_ansatz_set
    ap_ansatz_add = app.ap_ansatz_add
    ap_ansatz_mul = app.ap_ansatz_mul
    ap_ansatz_show = app.ap_ansatz_show
    ap_ansatz_lock = app.ap_ansatz_lock
    ap_ansatz_unlock = app.ap_ansatz_unlock
    ap_run = app.ap_run
    ap_conf_create = app.ap_conf_create
    ap_conf_dump = app.ap_conf_dump
    ap_conf_load = app.ap_conf_load
    ap_hamiltonian = app.ap_hamiltonian

    gm_run_g = app.gm_run_g
    ap_run_g = app.ap_run_g
