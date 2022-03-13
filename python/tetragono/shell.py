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
import cmd
import pickle
import importlib
import TAT
from . import common_variable
from . import conversion
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice
from .gradient import gradient_descent
from .normalize import normalize_state
from .expand_sampling import expand_sampling_lattice_dimension


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
        if common_variable.mpi_rank == 0:
            self.intro = """Welcome to the Tetragono shell. Type help or ? to list commands.""" + self.license

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
        if common_variable.mpi_rank == 0:
            os.system(line)
        common_variable.mpi_comm.barrier()

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
        seed : int
            The new random seed.
        """
        config = Config(line)
        self.seed(*config.args, **config.kwargs)

    def seed(self, seed):
        TAT.random.seed(seed)

    def su_gm_create(self, line, lattice_type):
        config = Config(line)
        model = importlib.import_module(config.args[0])
        if len(config.args) == 2 and config.args[-1] == "help":
            print(model.create.__doc__.replace("\n", "\n    "))
            return
        else:
            state = lattice_type(model.create(*config.args[1:], **config.kwargs))

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
        state = self.su_gm_create(line, SimpleUpdateLattice)
        if state != None:
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
        if common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.su, file)
        common_variable.mpi_comm.barrier()

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
        with open(name, "rb") as file:
            self.su = pickle.load(file)

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
            The new cut dimension used in simple update.
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
        common_variable.showln("Simple update lattice energy is", self.su.observe_energy())

    def do_su_to_ex(self, line):
        """
        Convert simple update lattice to exact lattice.
        """
        self.ex = conversion.simple_update_lattice_to_exact_state(self.su)

    def do_su_to_gm(self, line):
        """
        Convert simple update lattice to sampling lattice.
        """
        self.gm = conversion.simple_update_lattice_to_sampling_lattice(self.su)

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
        common_variable.showln("Exact state energy is", self.ex.observe_energy())

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
        if common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.ex, file)
        common_variable.mpi_comm.barrier()

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
        with open(name, "rb") as file:
            self.ex = pickle.load(file)

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
        state = self.su_gm_create(line, SamplingLattice)
        if state != None:
            self.gm = state

    def do_gm_normalize(self, line):
        """
        Normalize the lattice used for gradient method.

        Parameters
        ----------
        sampling_total_step : int
            The sampling total step.
        configuration_cut_dimension : int
            The cut dimension used in calculate configuration.
        direct_sampling_cut_dimension : int
            The cut dimension used in direct sampling.
        """
        config = Config(line)
        normalize_state(self.gm, *config.args, **config.kwargs)

    def do_gm_run(self, line):
        """
        Do gradient descent. see gradient.py for details.
        """
        config = Config(line)
        gradient_descent(self.gm, *config.args, **config.kwargs)

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
        if common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.gm, file)
        common_variable.mpi_comm.barrier()

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
        with open(name, "rb") as file:
            self.gm = pickle.load(file)

    def do_gm_expand(self, line):
        """
        Expand dimension of sampling lattice.

        Parameters
        ----------
        new_dimension : int
            The new dimension.
        epsilon : float
            The relative error added into tensor.
        """
        config = Config(line)
        expand_sampling_lattice_dimension(self.gm, *config.args, **config.kwargs)

    def do_gm_extend(self, line):
        print(" ###### DEPRECATE WARNING: gm_extend is deprecated, use gm_expand instead. ###### ")
        self.do_gm_expand(line)

    def do_gm_to_ex(self, line):
        """
        Convert sampling lattice to exact lattice.
        """
        self.ex = conversion.sampling_lattice_to_exact_state(self.gm)


class TetragonoScriptApp(TetragonoCommandApp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rawinput = False
        self.prompt = ""
        if common_variable.mpi_rank == 0:
            self.intro = """Welcome to the tetragono shell.""" + self.license

    def precmd(self, line):
        line = line.strip()
        line = super().precmd(line).strip()
        if line != "":
            common_variable.showln("TET> ", line, sep="")
        self.prompt = ""
        return line


if __name__ == "__main__":
    import os
    import sys
    help_message = """usage:
    shell.py
    shell.py [-h | -help | --help]
    shell.py script_file
    shell.py -- script"""
    # Tetragono Path
    if "TETPATH" in os.environ:
        pathes = os.environ["TETPATH"]
        for path in pathes.split(":"):
            sys.path.append(os.path.abspath(path))
    # Run
    if len(sys.argv) == 1:
        TetragonoCommandApp().cmdloop()
    elif len(sys.argv) == 2:
        script_file = sys.argv[1]
        if script_file in ["-h", "--help", "-help"]:
            common_variable.showln(help_message)
        else:
            with open(sys.argv[1], 'rt') as file:
                sys.path.append(os.path.dirname(os.path.abspath(sys.argv[1])))
                TetragonoScriptApp(stdin=file).cmdloop()
    elif sys.argv[1] == "--":
        commands = " ".join(sys.argv[2:]).replace(" - ", "\n")
        from io import StringIO
        file = StringIO(commands)
        TetragonoScriptApp(stdin=file).cmdloop()
    else:
        common_variable.showln("shell.py: Error: unrecognized command-line option")
        common_variable.showln(help_message)
        exit(1)
    common_variable.mpi_comm.barrier()
