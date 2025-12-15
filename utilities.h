#include <stdio.h>
#include <stdlib.h>

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