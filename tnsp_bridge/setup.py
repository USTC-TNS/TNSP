#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from setuptools import setup
from setuptools_scm import get_version

version = get_version(root="..")

try:
    with open("README.md", "rt", encoding="utf-8") as file:
        long_description = file.read()
except FileNotFoundError:
    long_description = "empty description"

setup(
    version=version,
    install_requires=[
        f"pytat=={version}",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
