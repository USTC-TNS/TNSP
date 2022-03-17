from setuptools import setup

version = "0.2.4"

setup(
    name="tetragono",
    version=version,
    description="OBC square tensor network state(PEPS) library",
    author="Hao Zhang",
    author_email="zh970205@mail.ustc.edu.cn",
    url="https://github.com/hzhangxyz/TAT",
    packages=["tetragono", "tetragono/common_variable"],
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
