CXX = g++
NVCC = nvcc
CXXFLAGS += -pg -g -O0 -std=c++11 -Wall -fprofile-arcs -ftest-coverage
CXXFLAGS += -lgomp hptt/lib/libhptt.a -Ihptt/include -g
CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -ldl -lpthread

cpu:
	$(CXX) TAT.cpp $(CXXFLAGS) $(CPUFLAGS)
