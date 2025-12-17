#include <cuda_runtime.h>
#include <stdio.h>
extern "C" {
#include "../utilities.h"
}

__global__ void convolution_global_kernel(const unsigned char *d_input, unsigned char *d_output,
                                         const float *d_kernel, int width, int height,
                                         int input_channels, int output_channels,
                                         int kernel_height, int kernel_width, int max_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (x >= width || y >= height || oc >= output_channels)
        return;

    int offset_y = kernel_height / 2;
    int offset_x = kernel_width / 2;
    float sum = 0.0f;

    for (int ic = 0; ic < input_channels; ic++)
    {
        for (int ky = 0; ky < kernel_height; ky++)
        {
            int py = y + ky - offset_y;
            if (py < 0)
                py = 0;
            if (py >= height)
                py = height - 1;

            for (int kx = 0; kx < kernel_width; kx++)
            {
                int px = x + kx - offset_x;
                if (px < 0)
                    px = 0;
                if (px >= width)
                    px = width - 1;

                int input_idx = ic * (height * width) + py * width + px;
                int kernel_idx = oc * (input_channels * kernel_height * kernel_width) +
                                 ic * (kernel_height * kernel_width) +
                                 ky * kernel_width + kx;
                sum += (float)d_input[input_idx] * d_kernel[kernel_idx];
            }
        }
    }

    if (sum < 0.0f)
        sum = 0.0f;
    if (sum > (float)max_val)
        sum = (float)max_val;

    int output_idx = oc * (height * width) + y * width + x;
    d_output[output_idx] = (unsigned char)sum;
}

Image *apply_convolution(Image *img, Kernel *kernel, double *time_taken)
{
    if (img->channels != kernel->input_channels)
    {
        fprintf(stderr, "Error: Image has %d channels but kernel expects %d\n", img->channels, kernel->input_channels);
        return NULL;
    }

    Image *result = allocate_result_image(img, kernel);
    if (!result)
        return NULL;

    unsigned char *h_input = flatten_image_host(img);
    float *h_kernel = flatten_kernel_host(kernel);
    if (!h_input || !h_kernel)
    {
        free(h_input);
        free(h_kernel);
        free_image(result);
        return NULL;
    }

    int input_size = img->channels * img->height * img->width;
    int kernel_size = kernel->output_channels * kernel->input_channels * kernel->height * kernel->width;
    int output_size = result->channels * result->height * result->width;

    unsigned char *d_input = NULL;
    unsigned char *d_output = NULL;
    float *d_kernel = NULL;

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16, 1);
    dim3 grid((img->width + block.x - 1) / block.x,
              (img->height + block.y - 1) / block.y,
              kernel->output_channels);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    convolution_global_kernel<<<grid, block>>>(d_input, d_output, d_kernel,
                                              img->width, img->height,
                                              kernel->input_channels, kernel->output_channels,
                                              kernel->height, kernel->width,
                                              img->max_val);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *time_taken = ms / 1000.0;

    CUDA_CHECK(cudaGetLastError());

    unsigned char *h_output = (unsigned char *)malloc(output_size * sizeof(unsigned char));
    if (!h_output)
    {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        free(h_input);
        free(h_kernel);
        free_image(result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return NULL;
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    unflatten_to_image(result, h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main(int argc, char *argv[])
{
    return run_filter(argc, argv);
}
