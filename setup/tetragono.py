#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from subprocess import check_output

version = check_output(["git", "describe"]).decode("utf-8")
version = version.replace("\n", "").replace("v", "").replace("-", ".post", 1).replace("-", "+")

setup(
    name="tetragono",
    version=version,
    description="OBC square tensor network state(PEPS) library",
    author="Hao Zhang",
    author_email="zh970205@mail.ustc.edu.cn",
    url="https://github.com/hzhangxyz/TAT",
    packages=[
        "tetragono",
        "tetragono/common_tensor",
        "tetragono/sampling_tools",
        "tetragono/shell_commands",
        "tetragono/auxiliaries",
    ],
    package_dir={"": "python"},
    install_requires=[
        f"PyTAT=={version}",
        f"lazy_graph=={version}",
        "mpi4py",
        "numpy",
    ],
    license="GPLv3",
    python_requires=">=3.9",
)
