CXX = clang++

TAT_VERSION = $(shell git describe --tags)
CXXFLAGS += -DTAT_VERSION=\"$(TAT_VERSION)\"

CXXFLAGS += -g -std=c++11 -fdata-sections -ffunction-sections -Wl,--gc-sections
CXXFLAGS += -static-libgcc -static-libstdc++ -Wl,-Bstatic -ljemalloc_pic
CXXFLAGS += -Wl,-Bstatic -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group
CXXFLAGS += -Wl,-Bdynamic -lpthread -lm -ldl -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64
CXXFLAGS += -Wl,-Bstatic -lhptt -Lhptt/lib -Ihptt/include
CXXFLAGS += -Iargs

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -pg -O0 -Wall -Wextra #-fprofile-arcs -ftest-coverage
else
	CXXFLAGS += -DNDEBUG -Ofast -march=native
	ifeq ($(CXX), g++)
		CXXFLAGS += -fwhole-program
	endif
endif

all: style compile

compile: test.out Heisenberg_MPS_SU.out Heisenberg_PEPS_SU.out

style: test.cpp.style TAT.hpp.style Heisenberg_MPS_SU.cpp.style Heisenberg_PEPS_SU.cpp.style

%.out: %.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

%.style: %
	astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type $<
