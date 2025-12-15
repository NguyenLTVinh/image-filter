.PHONY: all run

all:
	nvcc -o filter-parallel filter-parallel.cu

run: all
	./filter-parallel images/goldy.ppm kernels/gaussian-blur-15.txt goldy-kernel-output.ppm