# Copyright (c) 2018 Stig Rune Sellevag
#
# This file is distributed under the MIT License. See the accompanying file
# LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
# and conditions.

# Find the Intel Math Kernel Library.
#
# The following variables are defined if MKL is found:
#
# MKL_FOUND        - System has MKL
# MKL_INCLUDE_DIRS - MKL include file directory
# MKL_LIBRARY_DIRS - MKL library file directory
# MKL_LIBRARIES    - The MKL libraries
#
# The environmental variable MKLROOT is used to find the library.
#
# Example usage:
#
# find_package(MKL)
# if(MKL_FOUND)
#	  include_directories(${MKL_INCLUDE_DIRS})
#	  link_directories(${MKL_LIBRARY_DIRS})
#     target_link_libraries(TARGET ${MKL_LIBRARIES})
# endif()
#
# Note: 
# - Currently, the Intel LP64 interface layer is used for Intel(R) 64
#   architecture.

if(WIN32)
	if(${CMAKE_CL_64} EQUAL 1)
		set(INT_LIB "mkl_intel_lp64.lib")
		set(SEQ_LIB "mkl_sequential.lib")
		set(COR_LIB "mkl_core.lib")
		set(OMP_LIB "")
		set(THR_LIB "")
		set(MAT_LIB "")
		set(LDL_LIB "")
	else()
		set(INT_LIB "mkl_intel_c.lib")
		set(SEQ_LIB "mkl_sequential.lib")
		set(COR_LIB "mkl_core.lib")
		set(OMP_LIB "")
		set(THR_LIB "")
		set(MAT_LIB "")
		set(LDL_LIB "")
	endif()
elseif(APPLE)
	set(INT_LIB "-lmkl_intel_lp64")
	set(SEQ_LIB "-lmkl_sequential")
	set(COR_LIB "-lmkl_core")
	set(OMP_LIB "")
	set(THR_LIB "-lpthread")
	set(MAT_LIB "-lm")
	set(LDL_LIB "-ldl")
else()
	set(INT_LIB "-lmkl_intel_lp64")
	set(SEQ_LIB "-lmkl_sequential")
	set(COR_LIB "-lmkl_core")
	set(OMP_LIB "")
	set(THR_LIB "-lpthread")
	set(MAT_LIB "-lm")
	set(LDL_LIB "-ldl")
endif()

message("MKLROOT=$ENV{MKLROOT}")
message("MKLROOT=${MKLROOT}")

find_path(MKL_INCLUDE_DIRS mkl.h HINTS $ENV{MKLROOT}/include ${MKLROOT}/include)
if(WIN32)
	message("${CMAKE_SIZEOF_VOID_P}")
	if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
		message("-- finding intel64")
		find_path(MKL_LIBRARY_DIRS mkl_core.lib HINTS $ENV{MKLROOT}/lib/intel64 ${MKLROOT}/lib/intel64)
	else()
		message("-- find ia32")
		find_path(MKL_LIBRARY_DIRS mkl_core.lib HINTS $ENV{MKLROOT}/lib/ia32 ${MKLROOT}/lib/ia32)
	endif()
else()
	if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
		message("-- finding intel64")
		find_path(MKL_LIBRARY_DIRS libmkl_core.a HINTS $ENV{MKLROOT}/lib/intel64 ${MKLROOT}/lib/intel64)
	else()
		message("-- find ia32")
		find_path(MKL_LIBRARY_DIRS libmkl_core.a HINTS $ENV{MKLROOT}/lib/ia32 ${MKLROOT}/lib/ia32)
	endif()
endif()

message("MKL_INCLUDE_DIRS=${MKL_INCLUDE_DIRS}")
message("MKL_LIBRARY_DIRS=${MKL_LIBRARY_DIRS}")

if(MKL_INCLUDE_DIRS AND MKL_LIBRARY_DIRS)
    set(MKL_LIBRARIES ${INT_LIB} ${SEQ_LIB} ${COR_LIB} ${OMP_LIB} ${THR_LIB} ${MAT_LIB} ${LDL_LIB})
    set(MKL_FOUND ON)
	message("-- Intel MKL found")
else()
	message("-- Intel MKL not found")
endif()
