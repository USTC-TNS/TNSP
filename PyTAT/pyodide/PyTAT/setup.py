import os.path
from glob import glob
from setuptools import setup, Extension

ext_modules = [
    Extension("TAT",
              sorted([*glob(os.path.join("PyTAT", "*.cpp")), *glob(os.path.join("PyTAT", "generated_code", "*.cpp"))]),
              include_dirs=["include", "pybind11/include"],
              language="c++",
              extra_compile_args=["-std=c++17", "-g0"]),
]

setup(name="PyTAT", version="0.2.0", ext_modules=ext_modules)
