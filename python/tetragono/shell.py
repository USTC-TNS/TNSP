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

import os
import sys
import cmd
import importlib
import TAT
from .common_toolkit import (mpi_rank, mpi_size, mpi_comm, write_to_file, read_from_file, show, showln, seed_differ,
                             get_imported_function)
from . import conversion
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice, Configuration
from .gradient import gradient_descent


class Config():

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


class TetragonoCommandApp(cmd.Cmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = "TET> "
        self.stored_line = ""
        self.license = """
Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
        if mpi_rank == 0:
            self.intro = """Welcome to the Tetragono shell. Type help or ? to list commands.""" + self.license

        self.su = None
        self.ex = None
        self.gm = None
        self.gm_conf = []

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

    def do_seed(self, line):
        """
        Set random seed.

        Parameters
        ----------
        random_seed : int
            The new random seed.
        """
        config = Config(line)
        self.seed(*config.args, **config.kwargs)

    def seed(self, random_seed):
        TAT.random.seed(random_seed)

    @staticmethod
    def su_gm_create(lattice_type, *args, **kwargs):
        model = importlib.import_module(args[0])
        if len(args) == 2 and args[-1] == "help":
            print(model.create.__doc__.replace("\n", "\n    "))
            return None
        else:
            state = lattice_type(model.create(*args[1:], **kwargs))

        # pre normalize the tensor
        for l1 in range(state.L1):
            for l2 in range(state.L2):
                state[l1, l2] /= state[l1, l2].norm_max()
        return state

    def do_su_create(self, line):
        """
        Create a lattice used for simple update.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        config = Config(line)
        self.su_create(*config.args, **config.kwargs)

    def su_create(self, *args, **kwargs):
        state = self.su_gm_create(SimpleUpdateLattice, *args, **kwargs)
        if state is not None:
            self.su = state

    def do_su_dump(self, line):
        """
        Dump the simple update lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.su_dump(*config.args, **config.kwargs)

    def su_dump(self, name):
        write_to_file(self.su, name)

    def do_su_load(self, line):
        """
        Load the simple update lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.su_load(*config.args, **config.kwargs)

    def su_load(self, name):
        self.su = read_from_file(name)

    def do_su_update(self, line):
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
        config = Config(line)
        self.su_update(*config.args, **config.kwargs)

    def su_update(self, total_step, delta_tau, new_dimension):
        self.su.update(total_step, delta_tau, new_dimension)

    def do_su_energy(self, line):
        """
        Calculate simple update lattice with double layer auxiliaries.

        Parameters
        ----------
        cut_dimension : int
            The cut_dimension used in double layer auxiliaries.
        """
        config = Config(line)
        self.su_energy(*config.args, **config.kwargs)

    def su_energy(self, cut_dimension):
        self.su.initialize_auxiliaries(cut_dimension)
        showln("Simple update lattice energy is", self.su.observe_energy())

    def do_su_to_ex(self, line):
        """
        Convert simple update lattice to exact lattice.
        """
        config = Config(line)
        self.su_to_ex(*config.args, **config.kwargs)

    def su_to_ex(self):
        self.ex = conversion.simple_update_lattice_to_exact_state(self.su)

    def do_su_to_gm(self, line):
        """
        Convert simple update lattice to sampling lattice.
        """
        config = Config(line)
        self.su_to_gm(*config.args, **config.kwargs)

    def su_to_gm(self):
        self.gm = conversion.simple_update_lattice_to_sampling_lattice(self.su)
        self.gm_conf = []

    def do_ex_update(self, line):
        """
        Do exact update.

        Parameters
        ----------
        total_step : int
            The update total step to do.
        approximate_energy : float
            The approximate energy per site, it should ensure the ground state energy is the largest after shifting.
        """
        config = Config(line)
        self.ex_update(*config.args, **config.kwargs)

    def ex_update(self, total_step, approximate_energy):
        self.ex.update(total_step, approximate_energy)

    def do_ex_energy(self, line):
        """
        Calculate exact energy.
        """
        config = Config(line)
        self.ex_energy(*config.args, **config.kwargs)

    def ex_energy(self):
        showln("Exact state energy is", self.ex.observe_energy())

    def do_ex_dump(self, line):
        """
        Dump the exact update lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.ex_dump(*config.args, **config.kwargs)

    def ex_dump(self, name):
        write_to_file(self.ex, name)

    def do_ex_load(self, line):
        """
        Load the exact update lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.ex_load(*config.args, **config.kwargs)

    def ex_load(self, name):
        self.ex = read_from_file(name)

    def do_gm_create(self, line):
        """
        Create a lattice used for gradient method.

        Parameters
        ----------
        model : str
            The model names.
        args, kwargs
            Arguments passed to model creater function.
        """
        config = Config(line)
        self.gm_create(*config.args, **config.kwargs)

    def gm_create(self, *args, **kwargs):
        state = self.su_gm_create(SamplingLattice, *args, **kwargs)
        if state is not None:
            self.gm = state
            self.gm_conf = []

    def do_gm_run(self, line):
        """
        Do gradient descent. see gradient.py for details.
        """
        config = Config(line)
        self.gm_run(*config.args, **config.kwargs)

    def gm_run(self, *args, **kwargs):
        gradient_descent(self.gm, *args, **kwargs, sampling_configurations=self.gm_conf)

    def do_gm_dump(self, line):
        """
        Dump the sampling lattice into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.gm_dump(*config.args, **config.kwargs)

    def gm_dump(self, name):
        write_to_file(self.gm, name)

    def do_gm_conf_dump(self, line):
        """
        Dump the sampling lattice configuration into file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.gm_conf_dump(*config.args, **config.kwargs)

    def gm_conf_dump(self, name):
        write_to_file(self.gm_conf, name)

    def do_gm_load(self, line):
        """
        Load the sampling lattice from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.gm_load(*config.args, **config.kwargs)

    def gm_load(self, name):
        self.gm = read_from_file(name)
        self.gm_conf = []

    def do_gm_conf_load(self, line):
        """
        Load the sampling lattice configuration from file.

        Parameters
        ----------
        name : str
            The file name.
        """
        config = Config(line)
        self.gm_conf_load(*config.args, **config.kwargs)

    def gm_conf_load(self, name):
        self.gm_conf = read_from_file(name)

    def do_gm_conf_create(self, line):
        """
        Create configuration of sampling lattice.

        Parameters
        ----------
        module_name : str
            The module name to create initial configuration of sampling lattice.
        """
        config = Config(line)
        self.gm_conf_create(*config.args, **config.kwargs)

    def gm_conf_create(self, module_name):
        with seed_differ:
            # This configuration should never be used, so cut dimension is -1
            configuration = Configuration(self.gm, -1)
            configuration = get_imported_function(module_name, "initial_configuration")(configuration)
            self.gm_conf = mpi_comm.gather(configuration._configuration)

    def do_gm_expand(self, line):
        """
        Expand dimension of sampling lattice.

        Parameters
        ----------
        new_dimension : int | float
            The new dimension, or the amplitude of dimension expandance.
        epsilon : float
            The relative error added into tensor.
        """
        config = Config(line)
        self.gm_expand(*config.args, **config.kwargs)

    def gm_expand(self, new_dimension, epsilon):
        self.gm.expand_dimension(new_dimension, epsilon)

    def do_gm_to_ex(self, line):
        """
        Convert sampling lattice to exact lattice.
        """
        config = Config(line)
        self.gm_to_ex(*config.args, **config.kwargs)

    def gm_to_ex(self):
        self.ex = conversion.sampling_lattice_to_exact_state(self.gm)


class TetragonoScriptApp(TetragonoCommandApp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rawinput = False
        self.prompt = ""
        if mpi_rank == 0:
            self.intro = """Welcome to the tetragono shell.""" + self.license

    def precmd(self, line):
        line = line.strip()
        line = super().precmd(line).strip()
        if line != "":
            showln("TET> ", line, sep="")
        self.prompt = ""
        return line


# Tetragono Path
if "TETPATH" in os.environ:
    pathes = os.environ["TETPATH"]
    for path in pathes.split(":"):
        sys.path.append(os.path.abspath(path))
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
        from io import StringIO
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

    su_create = app.su_create
    su_dump = app.su_dump
    su_load = app.su_load
    su_update = app.su_update
    su_energy = app.su_energy
    su_to_ex = app.su_to_ex
    su_to_gm = app.su_to_gm

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
