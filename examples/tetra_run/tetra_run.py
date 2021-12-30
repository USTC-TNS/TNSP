#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import cmd
import pickle
import TAT
import tetragono as tet


class TetragonoCommandApp(cmd.Cmd):
    license = """
Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    """

    intro = """Welcome to the Tetragono shell. Type help or ? to list commands.""" + license

    prompt = "TET> "

    stored_line = ""

    def _int_float_str(self, i):
        try:
            return int(i)
        except ValueError:
            pass
        try:
            return float(i)
        except ValueError:
            pass
        return i

    def _parse(self, line, kw=False):
        split = line.split()
        args = [i for i in split if "=" not in i]
        kwargs = [i.split("=") for i in split if "=" in i]
        result = [self._int_float_str(i) for i in args]
        if kw:
            kv = {k: self._int_float_str(v) for k, v in kwargs}
            result.append(kv)
        else:
            if kwargs:
                raise ValueError
        return tuple(result)

    def precmd(self, line):
        if len(line) == 0:
            return line
        if line[-1] == "\\":
            self.stored_line = self.stored_line + " " + line[:-1]
            return ""
        line = self.stored_line + " " + line
        self.stored_line = ""
        line = line.split("#")[0]
        return line

    def emptyline(self):
        pass

    def do_EOF(self, line):
        () = self._parse(line)
        return True

    def do_seed(self, line):
        seed, = self._parse(line)
        TAT.random.seed(seed)

    def do_su_create(self, line):
        args = self._parse(line, True)
        model = __import__(args[0])
        kwargs = args[-1]
        args = args[1:-1]
        self.su = model.create(*args, **kwargs)

    def do_su_dump(self, line):
        name, = self._parse(line)
        if tet.common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.su, file)

    def do_su_load(self, line):
        name, = self._parse(line)
        with open(name, "rb") as file:
            self.su = pickle.load(file)

    def do_su_update(self, line):
        total_step, delta_tau, new_dimension = self._parse(line)
        self.su.update(total_step, delta_tau, new_dimension)

    def do_su_energy(self, line):
        cut_dimension, = self._parse(line)
        self.su.initialize_auxiliaries(cut_dimension)
        tet.common_variable.showln("Simple update lattice energy is", self.su.observe_energy())

    def do_su_to_ex(self, line):
        () = self._parse(line)
        self.ex = tet.conversion.simple_update_lattice_to_exact_state(self.su)

    def do_su_to_gm(self, line):
        cut_dimension, = self._parse(line)
        self.gm = tet.conversion.simple_update_lattice_to_sampling_lattice(self.su, cut_dimension)

    def do_ex_update(self, line):
        total_step, approximate_energy = self._parse(line)
        self.ex.update(total_step, approximate_energy)

    def do_ex_energy(self, line):
        () = self._parse(line)
        tet.common_variable.showln("Exact state energy is", self.ex.observe_energy())

    def do_ex_dump(self, line):
        name, = self._parse(line)
        if tet.common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.ex, file)

    def do_ex_load(self, line):
        name, = self._parse(line)
        with open(name, "rb") as file:
            self.ex = pickle.load(file)

    def do_gm_cut(self, line):
        cut, = self._parse(line)
        self.gm.cut_dimension = cut

    def do_gm_run(self, line):
        sampling_total_step, grad_total_step, grad_step_size, kv = self._parse(line, kw=True)

        config = type("Config", (object,), {})()

        config.sampling_total_step = sampling_total_step
        config.grad_total_step = grad_total_step
        config.grad_step_size = grad_step_size
        config.sampling_method = kv.get("sampling_method", "direct")
        config.log_file = kv.get("log_file", None)
        config.direct_sampling_cut_dimension = 4
        config.conjugate_gradient_method_step = 20
        config.metric_inverse_epsilon = 0.01
        config.use_gradient = grad_total_step != 0
        config.use_natural_gradient = kv.get("use_natural_gradient", 0) == 1
        config.use_line_search = kv.get("use_line_search", 0) == 1
        config.save_state_file = lambda x: None
        tet.gradient_descent(self.gm, config)

    def do_gm_dump(self, line):
        name, = self._parse(line)
        if tet.common_variable.mpi_rank == 0:
            with open(name, "wb") as file:
                pickle.dump(self.gm, file)

    def do_gm_load(self, line):
        name, = self._parse(line)
        with open(name, "rb") as file:
            self.gm = pickle.load(file)


class TetragonoScriptApp(TetragonoCommandApp):
    intro = """Welcome to the Tetragono shell.""" + TetragonoCommandApp.license
    use_rawinput = False
    prompt = ""

    def precmd(self, line):
        line = super().precmd(line)
        if line.replace(" ", "") != "":
            tet.common_variable.showln(super().prompt, line.strip(), sep="")
        return line


override_intro = None
if tet.common_variable.mpi_rank != 0:
    override_intro = ""

if __name__ == "__main__":
    import sys
    help_message = "Usage: tetra_run.py [script_file]"
    if len(sys.argv) == 1:
        TetragonoCommandApp().cmdloop(intro=override_intro)
    elif len(sys.argv) == 2:
        script_file = sys.argv[1]
        if script_file in ["-h", "--help", "-help"]:
            tet.common_variable.showln(help_message)
        else:
            with open(sys.argv[1], 'rt') as file:
                TetragonoScriptApp(stdin=file).cmdloop(intro=override_intro)
    elif sys.argv[1] == "--":
        commands = " ".join(sys.argv[2:]).replace("-", "\n")
        from io import StringIO
        file = StringIO(commands)
        TetragonoScriptApp(stdin=file).cmdloop(intro=override_intro)
    else:
        tet.common_variable.showln("tetra_run: Error: unrecognized command-line option")
        tet.common_variable.showln(help_message)
        exit(1)
