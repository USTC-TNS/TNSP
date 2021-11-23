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

import os
import fire
import pickle
import TAT
import tetragono as tet


class App:
    __slots__ = ["_file_name", "_state"]

    def attr(self):
        return self._state

    def __init__(self, file_name):
        self._file_name = file_name
        if os.path.isfile(self._file_name):
            with open(self._file_name, "rb") as file:
                self._state = pickle.load(file)
            self._retype(self.type())

    def _retype(self, name):
        self.__class__ = globals()[name + "App"]
        return self

    def new(self, *args, **kwargs):
        """
        Create new state
        """
        self._state = self._create(*args, **kwargs)
        self.save()
        self._retype("SimpleUpdateLattice")
        return self

    def file(self, file_name):
        """
        Set the file name where the state will be saved
        """
        self._file_name = file_name
        return self

    def save(self):
        """
        Save the file explicitly,
        otherwise state will only be saved when simple updating and gradient descent,
        which is desired to take a long time to run
        """
        print(" Saving to", self._file_name)
        with open(self._file_name, "wb") as file:
            pickle.dump(self._state, file)
        return self

    def type(self):
        """
        Get the current type of the state
        """
        return type(self._state).__name__

    def end(self):
        pass


class ExactStateApp(App):
    __slots__ = []

    def update(self, total_step, approximate_energy=-0.5):
        """
        Exact update
        """
        self._state.update(total_step, approximate_energy)
        return self

    def energy(self):
        """
        Observe energy of exact state
        """
        print(" Exact state energy is", self._state.observe_energy())
        return self


class SimpleUpdateLatticeApp(App):
    __slots__ = []

    def update(self, total_step, delta_tau, new_dimension):
        """
        Simple Update
        """
        self._state.update(total_step, delta_tau, new_dimension)
        self.save()
        return self

    def exact(self):
        """
        Convert the state to exact state
        """
        print(" Converting to exact state")
        self._state = tet.conversion.simple_update_lattice_to_exact_state(self._state)
        self._retype("ExactState")
        return self

    def sampling(self, cut_dimension):
        """
        Convert the state to sampling lattice
        """
        print(" Converting to sampling lattice with Dc", cut_dimension)
        self._state = tet.conversion.simple_update_lattice_to_sampling_lattice(self._state, cut_dimension)
        self._retype("SamplingLattice")
        return self


class SamplingLatticeApp(App):
    __slots__ = []

    def cut(self, cut):
        """
        Change cut dimension
        """
        print(" Change cut dimension to", cut)
        self._state.cut_dimension = cut
        return self

    def configuration(self):
        """
        Initialize configuration
        """
        self._configuration(self._state)
        return self

    def run(self, total_step, grad_total_step=1, grad_step_size=0):
        state = self._state

        sampling = tet.SweepSampling(state)
        observer = tet.Observer(state)
        observer.add_energy()
        if grad_step_size != 0:
            observer.enable_gradient()
        for grad_step in range(grad_total_step):
            observer.flush()
            for step in range(total_step):
                observer(sampling())
                print(tet.common_variable.clear_line, f"sampling, {total_step=}, energy={observer.energy}, {step=}", end="\r")
            if grad_step_size != 0:
                print(tet.common_variable.clear_line, f"grad {grad_step}/{grad_total_step}, step_size={grad_step_size}, sampling={total_step}, energy={observer.energy}")
                grad = observer.gradient
                for i in range(state.L1):
                    for j in range(state.L2):
                        state[i, j] -= grad_step_size * grad[i][j]
                state.configuration.refresh_all()
                self.save()
            else:
                print(tet.common_variable.clear_line, f"sampling done, {total_step=}, energy={observer.energy}")
        return self


def get_app(model, file_name, seed=None):
    if seed is not None:
        print(" Set the random seed to", seed)
        TAT.random.seed(seed)
    mod = __import__(model)
    if hasattr(mod, "configuration"):
        SamplingLatticeApp._configuration = staticmethod(mod.configuration)
    if hasattr(mod, "create"):
        App._create = staticmethod(mod.create)
    return App(file_name)


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(get_app)
