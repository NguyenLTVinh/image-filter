.PHONY: all run

all:
	gcc -std=c11 -O3 filter-serial.c utilities.c -o filter-serial
	nvcc -O3 -arch=sm_75 filter-parallel.cu utilities.c -o filter-parallel
run: all
	./filter-parallel images/goldy.ppm kernels/gaussian-blur-15.txt goldy-kernel-output.ppm