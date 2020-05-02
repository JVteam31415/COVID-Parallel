all: covid-mpi.c covid.cu covidmovement.c
	mpicc -g covid-mpi.c -c -o covid-main.o
	nvcc -g -gencode arch=compute_70,code=sm_70 covid.cu -o a.out
	mpicc -g covid-main.o a.out -o covid-cuda-mpi-exe \
	-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++