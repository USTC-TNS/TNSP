CXX = g++
NVCC = nvcc
#CXXFLAGS += -g -O3
CPUFLAGS = -lpthread -lgomp -lcblas hptt/lib/libhptt.a -Ihptt/include -g
CUDAFLAGS = -lcublas cutt/lib/libcutt.a -Icutt/include -gencode arch=compute_50,code=sm_50 -G

cpu:
	$(CXX) TAT.cpp $(CXXFLAGS) $(CPUFLAGS)

cuda:
	$(NVCC) main.cu $(CXXFLAGS) $(CUDAFLAGS)
