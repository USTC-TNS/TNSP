CXX = clang++
STATIC ?= 0
DEBUG ?= 0
MKL=/opt/intel/compilers_and_libraries_2019.2.187/linux/mkl

TAT_VERSION = $(shell git describe --tags)
CXXFLAGS += -DTAT_VERSION=\"$(TAT_VERSION)\"

CXXFLAGS += -g -std=c++11 -fdata-sections -ffunction-sections -Wl,--gc-sections
ifeq ($(STATIC), 1)
	CXXFLAGS += -static-libgcc -static-libstdc++ -Wl,-Bstatic -ljemalloc_pic
	CXXFLAGS += -Wl,-Bstatic -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group
	CXXFLAGS += -Wl,-Bdynamic -lpthread -lm -ldl -I$(MKL)/include -L$(MKL)/lib/intel64
	CXXFLAGS += -Wl,-Bstatic -lhptt -Lhptt/lib -Ihptt/include
	CXXFLAGS += -Iargs -Irang/include
else
	CXXFLAGS += -ljemalloc
	CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	CXXFLAGS += -lpthread -lm -ldl -I$(MKL)/include -L$(MKL)/lib/intel64 -Wl,-rpath,$(MKL)/lib/intel64
	CXXFLAGS += -lhptt -Lhptt/lib -Ihptt/include -Wl,-rpath,./hptt/lib
	CXXFLAGS += -Iargs -Irang/include
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -pg -O0 -Wall -Wextra #-fprofile-arcs -ftest-coverage
else
	CXXFLAGS += -DNDEBUG -Ofast -march=native -Wall -Wextra
	ifeq ($(CXX), g++)
		CXXFLAGS += -fwhole-program
	endif
endif

HEADER=$(wildcard ./*.hpp)
HEADER_STYLE=$(HEADER:.hpp=.hpp.style)
SRC=$(wildcard ./*.cpp)
SRC_STYLE=$(SRC:.cpp=.cpp.style)
COMPILE=$(SRC:.cpp=.out)
ORIG=$(SRC:.cpp=.cpp.orig) $(HEADER:.hpp=.hpp.orig)
MISC_TO_REMOVE=test_io2.out test_io.out

all: style compile

compile: $(COMPILE)

style: $(HEADER_STYLE) $(SRC_STYLE)

%.out: %.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

%.style: %
	astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type $<

clean:
	rm -rf ${COMPILE} ${ORIG} $(MISC_TO_REMOVE)
