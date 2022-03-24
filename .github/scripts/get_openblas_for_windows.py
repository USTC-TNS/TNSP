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

import urllib.request
from zipfile import ZipFile

openblas_url = "https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20-x64.zip"

urllib.request.urlretrieve(openblas_url, "openblas.zip")

with ZipFile("openblas.zip", "r") as zip_file:
    zip_file.extract("lib/libopenblas.lib", ".")
