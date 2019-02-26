cpu:
	clang++ main.cpp -lpthread -lgomp -lcblas hptt/lib/libhptt.a -Ihptt/include

cuda:
	nvcc main.cu -lcublas cutt/lib/libcutt.a -Icutt/include -gencode arch=compute_50,code=sm_50
