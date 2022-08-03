#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from .open_string import OpenString
from .closed_string import ClosedString
from .periodic_string import PeriodicString
from .product_ansatz import ProductAnsatz
try:
    import torch
except ModuleNotFoundError:

    class ConvolutionalNeural:

        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch needed for ConvolutionalNeural")
else:
    from .convolutional_neural import ConvolutionalNeural
