CXX ?= g++
NVCC = nvcc
CXXFLAGS += -g -std=c++11 -fdata-sections -ffunction-sections -Wl,--gc-sections
CXXFLAGS += -static-libgcc -static-libstdc++ -Wl,-Bstatic -ljemalloc_pic
CXXFLAGS += -Wl,-Bstatic -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -Wl,-Bdynamic -lpthread -lm -ldl
CXXFLAGS += -Wl,-Bdynamic -lgomp -Wl,-Bstatic -lhptt -Lhptt/lib -Ihptt/include

DEBUG ?= 1
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -pg -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage
else
	CXXFLAGS += -DNDEBUG -Ofast
endif

FILE ?= TAT.cpp
all:
	$(CXX) $(FILE) $(CXXFLAGS)

SFILE ?= TAT.hpp
style:
	astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type $(SFILE)
