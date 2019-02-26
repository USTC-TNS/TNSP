cpu:
	clang++ main.cpp -lpthread -lgomp -lcblas hptt/lib/libhptt.a -Ihptt/include

cuda:
	nvcc main.cu
