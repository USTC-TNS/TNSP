CXX = g++
NVCC = nvcc
CXXFLAGS += -std=c++11
CXXFLAGS += -lgomp hptt/lib/libhptt.a -Ihptt/include -g
CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -ldl -lpthread
DEBUG = -pg -g -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage
NDEBUG = -DNDEBUG -O3

debug:
	$(CXX) TAT.cpp $(CXXFLAGS) $(DEBUG)

ndebug:
	$(CXX) TAT.cpp $(CXXFLAGS) $(NDEBUG)
