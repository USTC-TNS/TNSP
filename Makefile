CXX ?= g++
NVCC = nvcc
CXXFLAGS += -std=c++11 -static-libstdc++ -static-libgcc -fdata-sections -ffunction-sections -Wl,--gc-sections
CXXFLAGS += -lgomp hptt/lib/libhptt.a -Ihptt/include
CXXFLAGS += -Wl,-Bstatic -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -Wl,-Bdynamic -lpthread -lm -ldl

DEBUG?=1
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -pg -g -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage
else
	CXXFLAGS += -DNDEBUG -Ofast
endif

all:
	$(CXX) TAT.cpp $(CXXFLAGS)
