CXX = clang++
NVCC = nvcc
CXXFLAGS += -g -O3
CPUFLAGS = -lpthread -lgomp -lcblas hppt/lib/libhptt.a -Ihptt/include
CUDAFLAGS = -lcublas cutt/lib/libcutt.a -Icutt/include -gencode arch=compute_50,code=sm_50

cpu:
	$(CXX) main.cpp $(CXXFLAGS) $(CPUFLAGS)

cuda:
	$(NVCC) main.cu $(CXXFLAGS) $(CUDAFLAGS)
