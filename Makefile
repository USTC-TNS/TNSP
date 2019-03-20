CXX = g++
NVCC = nvcc
CXXFLAGS += -pg -g -O0 -std=c++11 -Wall -fprofile-arcs -ftest-coverage
CXXFLAGS += -lpthread -lgomp -lcblas hptt/lib/libhptt.a -Ihptt/include -g

cpu:
	$(CXX) TAT.cpp $(CXXFLAGS) $(CPUFLAGS)
