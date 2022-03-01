from setuptools import setup

setup(
    name="tetragono",
    version="0.2.0",
    description="OBC square tensor network state(PEPS) library",
    author="Hao Zhang",
    author_email="zh970205@mail.ustc.edu.cn",
    url="https://github.com/hzhangxyz/TAT",
    packages=["tetragono", "tetragono/common_variable"],
    package_dir={"": "python"},
    install_requires=[
        "PyTAT",
        "lazy_graph",
        "mpi4py",
        "numpy",
    ],
    license="GPLv3",
)
