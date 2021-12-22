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

    def _parse(self, line):
        return tuple(self._int_float_str(i) for i in line.split())

    def precmd(self, line):
        return line.split("#")[0]

    def emptyline(self):
        pass

    def do_EOF(self, line):
        () = self._parse(line)
        return True

    def do_seed(self, line):
        seed, = self._parse(line)
        TAT.random.seed(seed)

    def do_model_set(self, line):
        name, = self._parse(line)
        self.model = __import__(name)

    def do_su_create(self, line):
        args = self._parse(line)
        self.su = self.model.create(*args)

    def do_su_dump(self, line):
        name, = self._parse(line)
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
        print(" Simple update lattice energy is", self.su.observe_energy())

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
        print(" Exact state energy is", self.ex.observe_energy())

    def do_ex_dump(self, line):
        name, = self._parse(line)
        with open(name, "wb") as file:
            pickle.dump(self.ex, file)

    def do_ex_load(self, line):
        name, = self._parse(line)
        with open(name, "rb") as file:
            self.ex = pickle.load(file)

    def do_gm_cut(self, line):
        cut, = self._parse(line)
        self.gm.cut_dimension = cut

    def do_gm_cfg_create(self, line):
        () = self._parse(line)
        self.model.configuration(self.gm)

    def do_gm_cfg_ensure(self, line):
        () = self._parse(line)
        if not self.gm.configuration.valid():
            self.model.configuration(self.gm)

    def do_gm_run(self, line):
        total_step, grad_total_step, grad_step_size, log_file = self._parse(line)
        state = self.gm

        direct_sampling_cut_dimension = 4
        conjugate_gradient_method_step = 20

        sampling = tet.DirectSampling(state, direct_sampling_cut_dimension)
        observer = tet.Observer(state)
        observer.add_energy()
        if grad_step_size != 0:
            observer.enable_gradient()
            observer.enable_natural_gradient()
        for grad_step in range(grad_total_step):
            observer.flush()
            for step in range(total_step):
                observer(sampling())
                print(tet.common_variable.clear_line,
                      f"sampling, {total_step=}, energy={observer.energy}, {step=}",
                      end="\r")
            if grad_step_size != 0:
                with open(log_file, "a") as file:
                    print(*observer.energy, file=file)
                print(
                    tet.common_variable.clear_line,
                    f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}, sampling={total_step}, energy={observer.energy}"
                )
                grad = observer.natural_gradient(conjugate_gradient_method_step)
                for i in range(state.L1):
                    for j in range(state.L2):
                        state[i, j] -= grad_step_size * grad[i][j]
                state.configuration.refresh_all()
                sampling.refresh_all()
            else:
                print(tet.common_variable.clear_line, f"sampling done, {total_step=}, energy={observer.energy}")

    def do_gm_dump(self, line):
        name, = self._parse(line)
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
        if line != "":
            print(super().prompt, line, sep="")
        return line


if __name__ == "__main__":
    import sys
    help_message = "Usage: tetra_run.py [script_file]"
    if len(sys.argv) == 1:
        TetragonoCommandApp().cmdloop()
    elif len(sys.argv) == 2:
        script_file = sys.argv[1]
        if script_file in ["-h", "--help", "-help"]:
            print(help_message)
        else:
            with open(sys.argv[1], 'rt') as file:
                TetragonoScriptApp(stdin=file).cmdloop()
    elif sys.argv[1] == "--":
        commands = " ".join(sys.argv[2:]).replace("-", "\n")
        from io import StringIO
        file = StringIO(commands)
        TetragonoScriptApp(stdin=file).cmdloop()
    else:
        print("tetra_run: Error: unrecognized command-line option")
        print(help_message)
        exit(1)
