all: covid.cu covid-mpi.c
	mpicc -g covid-mpi.c -c -o covid-mpi.o
	nvcc -g -G -arch=sm_70 covid.cu -c -o covid-cuda.o
	mpicc -g covid-mpi.o covid-cuda.o -o covid-cuda-mpi.out -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
