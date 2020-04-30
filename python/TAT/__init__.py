import cppyy
cppyy.load_library("blas")
cppyy.load_library("lapack")
cppyy.include("../include/TAT/TAT.hpp")
std = cppyy.gbl.std
TAT = cppyy.gbl.TAT
