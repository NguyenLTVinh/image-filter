.PHONY: all run

all:
	nvcc -o filter filter.cu

run: all
	./filter images/goldy.ppm kernels/blur.txt goldy-blur-output.ppm