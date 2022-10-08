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

from setuptools import setup, find_packages
from subprocess import check_output

version = check_output(["git", "describe"]).decode("utf-8")
version = version.replace("\n", "").replace("v", "").replace("-", ".post", 1).replace("-", "+")

try:
    with open("README.md", "rt", encoding="utf-8") as file:
        long_description = file.read()
except FileNotFoundError:
    long_description = "empty description"

setup(
    name="wrapper_TAT",
    version=version,
    description="wrapper for other tensor library to TAT interface",
    author="Hao Zhang",
    author_email="zh970205@mail.ustc.edu.cn",
    url="https://github.com/hzhangxyz/TAT/tree/TAT/wrapper-TAT",
    packages=find_packages(),
    install_requires=[
        f"PyTAT=={version}",
        "multimethod",
        "numpy",
        "torch",
    ],
    license="GPLv3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)