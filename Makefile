all: covid.cu
	nvcc -g -gencode arch=compute_70,code=sm_70 covid.cu -o a.out
