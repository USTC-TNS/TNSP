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

import os
import sys
import pathlib
from subprocess import check_output
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_original

version = check_output(["git", "describe"]).decode("utf-8")
version = version.replace("\n", "").replace("v", "").replace("-", ".post", 1).replace("-", "+")


class CMakeExtension(Extension):

    def __init__(self, name, sources=[]):
        super().__init__(name=name, sources=sources)


class build_ext(build_ext_original):

    def run(self):
        for extension in self.extensions:
            self.build_cmake(extension)

    def build_cmake(self, extension):
        cwd = pathlib.Path().absolute()
        build_dir = pathlib.Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        extension_dir = pathlib.Path(self.get_ext_fullpath(extension.name)).parent
        extension_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extension_dir.absolute()),
            "-DCMAKE_BUILD_TYPE=" + "Release",
            "-DTAT_USE_MPI=" + "OFF",
            "-DTAT_FORCE_VERSION=" + version,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]
        if "CMAKEFLAGS" in os.environ:
            cmake_args += os.environ["CMAKEFLAGS"].split("|")
        os.chdir(str(build_dir))
        self.spawn(['cmake', str(cwd)] + cmake_args)

        if not self.dry_run:
            if "MAKEFLAGS" in os.environ:
                make_args = os.environ["MAKEFLAGS"].split("|")
            else:
                make_args = []
            self.spawn(["cmake", "--build", ".", "--target", extension.name] + make_args)
        os.chdir(str(cwd))


setup(
    name="PyTAT",
    version=version,
    description="python binding for TAT(TAT is A Tensor library)",
    author="Hao Zhang",
    author_email="zh970205@mail.ustc.edu.cn",
    url="https://github.com/hzhangxyz/TAT",
    ext_modules=[CMakeExtension("PyTAT")],
    cmdclass={
        'build_ext': build_ext,
    },
    install_requires=[
        "numpy",
    ],
    license="GPLv3",
)
