import os.path
from glob import glob
from distutils.core import setup, Extension

ext_modules = [
    Extension("TAT",
              sorted([*glob(os.path.join("PyTAT", "*.cpp")), *glob(os.path.join("PyTAT", "generated_code", "*.cpp"))]),
              include_dirs=["include", "pybind11/include"],
              define_macros=[("TAT_VERSION", "\"unknown\""), ("TAT_BUILD_TYPE", "\"unknown\""), ("TAT_COMPILER_INFORMATION", "\"unknown\"")],
              language="c++",
              extra_compile_args=["-std=c++17"]),
]

setup(name="TAT", version="0.1.4", ext_modules=ext_modules)
