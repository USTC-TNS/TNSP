import os
import sys
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_original


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
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]
        os.chdir(str(build_dir))
        self.spawn(['cmake', str(cwd)] + cmake_args)

        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--target", extension.name, "--parallel", "4"])
        os.chdir(str(cwd))


setup(
    name="PyTAT",
    version="0.2.3",
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
