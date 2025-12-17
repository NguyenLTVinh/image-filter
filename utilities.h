#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef struct
{
    int width;
    int height;
    int channels;
    int max_val;
    unsigned char ***data; // [channel][height][width]
} Image;

typedef struct
{
    int height;
    int width;
    int input_channels;
    int output_channels;
    float ****data; // [out_ch][in_ch][ky][kx]
} Kernel;

typedef struct
{
    int num_kernels;
    Kernel **kernels;
} KernelSet;

Image *read_image(const char *filename);
int write_image(const char *filename, Image *img);
void free_image(Image *img);
KernelSet *read_kernels(const char *filename);
void free_kernel_set(KernelSet *kset);

Image *apply_convolution(Image *img, Kernel *kernel, double *time_taken);
int run_filter(int argc, char *argv[]);

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
    
unsigned char *flatten_image_host(const Image *img);
float *flatten_kernel_host(const Kernel *kernel);
Image *allocate_result_image(const Image *img, const Kernel *kernel);
void unflatten_to_image(Image *result, const unsigned char *flat);

// Directory utilities
char **get_image_files(const char *directory, int *count);
void free_string_array(char **array, int count);
int create_directory(const char *path);
int is_image_file(const char *filename);
char *get_filename(const char *path);
char *join_path(const char *dir, const char *filename);
int is_directory(const char *path);

#endif