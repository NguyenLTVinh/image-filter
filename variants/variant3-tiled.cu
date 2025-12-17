#include <cuda_runtime.h>
#include <stdio.h>
extern "C" {
#include "../utilities.h"
}

__global__ void convolution_tiled_kernel(const unsigned char *d_input, unsigned char *d_output,
                                          const float *d_kernel, int width, int height,
                                          int input_channels, int output_channels,
                                          int kernel_height, int kernel_width, int max_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    int pad_y = kernel_height / 2;
    int pad_x = kernel_width / 2;

    int tile_w = blockDim.x + 2 * pad_x;
    int tile_h = blockDim.y + 2 * pad_y;
    int channel_stride = tile_w * tile_h;

    extern __shared__ unsigned char tile[];

    for (int ic = 0; ic < input_channels; ic++)
    {
        for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y)
        {
            int global_y = blockIdx.y * blockDim.y + ty - pad_y;
            if (global_y < 0)
                global_y = 0;
            if (global_y >= height)
                global_y = height - 1;

            for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x)
            {
                int global_x = blockIdx.x * blockDim.x + tx - pad_x;
                if (global_x < 0)
                    global_x = 0;
                if (global_x >= width)
                    global_x = width - 1;

                int input_idx = ic * (height * width) + global_y * width + global_x;
                int shared_idx = ic * channel_stride + ty * tile_w + tx;
                tile[shared_idx] = d_input[input_idx];
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height || oc >= output_channels)
        return;

    float sum = 0.0f;
    for (int ic = 0; ic < input_channels; ic++)
    {
        int base = ic * channel_stride;
        for (int ky = 0; ky < kernel_height; ky++)
        {
            int tile_y = threadIdx.y + ky;
            for (int kx = 0; kx < kernel_width; kx++)
            {
                int tile_x = threadIdx.x + kx;
                unsigned char pixel_val = tile[base + tile_y * tile_w + tile_x];
                int kernel_idx = oc * (input_channels * kernel_height * kernel_width) +
                                 ic * (kernel_height * kernel_width) +
                                 ky * kernel_width + kx;
                sum += (float)pixel_val * d_kernel[kernel_idx];
            }
        }
    }

    if (sum < 0.0f)
        sum = 0.0f;
    if (sum > (float)max_val)
        sum = (float)max_val;

    int out_idx = oc * (height * width) + y * width + x;
    d_output[out_idx] = (unsigned char)sum;
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

    int pad_x = kernel->width / 2;
    int pad_y = kernel->height / 2;
    size_t shared = (block.x + 2 * pad_x) * (block.y + 2 * pad_y) * img->channels * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    convolution_tiled_kernel<<<grid, block, shared>>>(d_input, d_output, d_kernel,
                                                             img->width, img->height,
                                                             kernel->input_channels, kernel->output_channels,
                                                             kernel->height, kernel->width, img->max_val);

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
