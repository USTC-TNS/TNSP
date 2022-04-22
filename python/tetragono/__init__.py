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

# States
from .abstract_state import AbstractState
from .exact_state import ExactState
from .abstract_lattice import AbstractLattice
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_lattice import SamplingLattice, Configuration

# Miscellaneous
from . import conversion
from .sampling_tools import Observer, SweepSampling, ErgodicSampling, DirectSampling
from . import common_tensor
from . import common_toolkit
from .common_toolkit import *

# Deprecated
from . import common_variable_deprecated


def __getattr__(name):
    if name == "common_variable":
        print(
            " ###### DEPRECATE WARNING: common_variable is deprecated, use common_tensor or common_toolkit instead. ###### "
        )
        return common_variable_deprecated
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
