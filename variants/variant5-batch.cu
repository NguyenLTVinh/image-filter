#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
extern "C" {
#include "../utilities.h"
}

__global__ void convolution_batch_kernel(const unsigned char *d_input,
                                         unsigned char *d_output,
                                         const float *d_kernel,
                                         int width,
                                         int height,
                                         int input_channels,
                                         int output_channels,
                                         int kernel_height,
                                         int kernel_width,
                                         int batch_size,
                                         int max_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int combined = blockIdx.z;
    int b = combined / output_channels;
    int oc = combined - b * output_channels;

    if (b >= batch_size || oc >= output_channels || x >= width || y >= height)
        return;

    int offset_y = kernel_height / 2;
    int offset_x = kernel_width / 2;
    float sum = 0.0f;

    const int img_stride = input_channels * height * width;
    const int out_stride = output_channels * height * width;
    const int kernel_base_oc = oc * (input_channels * kernel_height * kernel_width);

    for (int ic = 0; ic < input_channels; ic++)
    {
        int ic_kernel_offset = kernel_base_oc + ic * (kernel_height * kernel_width);
        int img_channel_base = b * img_stride + ic * (height * width);

        for (int ky = 0; ky < kernel_height; ky++)
        {
            int py = y + ky - offset_y;
            py = max(0, min(py, height - 1));

            int row_base = img_channel_base + py * width;
            int krow_base = ic_kernel_offset + ky * kernel_width;

            for (int kx = 0; kx < kernel_width; kx++)
            {
                int px = x + kx - offset_x;
                px = max(0, min(px, width - 1));
                sum += (float)d_input[row_base + px] * d_kernel[krow_base + kx];
            }
        }
    }

    sum = fminf(fmaxf(sum, 0.0f), (float)max_val);

    int out_idx = b * out_stride + oc * (height * width) + y * width + x;
    d_output[out_idx] = (unsigned char)sum;
}

static void free_images(Image **imgs, int count)
{
    if (!imgs) return;
    for (int i = 0; i < count; i++)
        free_image(imgs[i]);
    free(imgs);
}

Image *apply_convolution(Image *img, Kernel *kernel, double *time_taken)
{
    if (time_taken) *time_taken = 0.0;
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <input file|dir> <kernel.txt> <output file|dir>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *kernel_path = argv[2];
    const char *output_path = argv[3];

    KernelSet *kset = read_kernels(kernel_path);
    if (!kset)
        return 1;
    if (kset->num_kernels < 1)
    {
        fprintf(stderr, "Error: kernel file must contain at least one kernel.\n");
        free_kernel_set(kset);
        return 1;
    }

    Kernel *kernel = kset->kernels[0];

    float *h_kernel = flatten_kernel_host(kernel);
    if (!h_kernel)
    {
        free_kernel_set(kset);
        return 1;
    }

    int kernel_elems = kernel->output_channels * kernel->input_channels * kernel->height * kernel->width;

    if (is_directory(input_path))
    {
        if (!create_directory(output_path))
        {
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        int num_images = 0;
        char **image_files = get_image_files(input_path, &num_images);
        if (!image_files || num_images == 0)
        {
            fprintf(stderr, "Error: no images found in %s\n", input_path);
            free_string_array(image_files, num_images);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        Image **imgs = (Image **)calloc(num_images, sizeof(Image *));
        if (!imgs)
        {
            free_string_array(image_files, num_images);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        int width = -1, height = -1, channels = -1;
        for (int i = 0; i < num_images; i++)
        {
            imgs[i] = read_image(image_files[i]);
            if (!imgs[i])
            {
                free_images(imgs, num_images);
                free_string_array(image_files, num_images);
                free(h_kernel);
                free_kernel_set(kset);
                return 1;
            }
            if (i == 0)
            {
                width = imgs[i]->width;
                height = imgs[i]->height;
                channels = imgs[i]->channels;
            }
            else if (imgs[i]->width != width || imgs[i]->height != height || imgs[i]->channels != channels)
            {
                fprintf(stderr, "Error: all images in batch must share dimensions/channels.\n");
                free_images(imgs, num_images);
                free_string_array(image_files, num_images);
                free(h_kernel);
                free_kernel_set(kset);
                return 1;
            }
        }

        if (channels != kernel->input_channels)
        {
            fprintf(stderr, "Error: images have %d channels, kernel expects %d.\n", channels, kernel->input_channels);
            free_images(imgs, num_images);
            free_string_array(image_files, num_images);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        int img_stride = channels * height * width;
        int out_stride = kernel->output_channels * height * width;
        size_t input_bytes = (size_t)num_images * img_stride;
        size_t output_bytes = (size_t)num_images * out_stride;

        unsigned char *h_input = (unsigned char *)malloc(input_bytes);
        unsigned char *h_output = (unsigned char *)malloc(output_bytes);
        if (!h_input || !h_output)
        {
            free(h_input);
            free(h_output);
            free_images(imgs, num_images);
            free_string_array(image_files, num_images);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        for (int i = 0; i < num_images; i++)
        {
            unsigned char *flat = flatten_image_host(imgs[i]);
            if (!flat)
            {
                free(h_input);
                free(h_output);
                free_images(imgs, num_images);
                free_string_array(image_files, num_images);
                free(h_kernel);
                free_kernel_set(kset);
                return 1;
            }
            memcpy(h_input + i * img_stride, flat, img_stride);
            free(flat);
        }

        unsigned char *d_input = NULL;
        unsigned char *d_output = NULL;
        float *d_kernel = NULL;
        CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_elems * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_elems * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16, 1);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y,
                  num_images * kernel->output_channels);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        convolution_batch_kernel<<<grid, block>>>(d_input, d_output,
                                                  d_kernel,
                                                  width, height,
                                                  kernel->input_channels, kernel->output_channels,
                                                  kernel->height, kernel->width,
                                                  num_images, imgs[0]->max_val);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double seconds = ms / 1000.0;
        printf("Total computation time: %.6f seconds\n", seconds);
        printf("Batch of %d images processed.\n", num_images);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_images; i++)
        {
            Image *res = allocate_result_image(imgs[i], kernel);
            if (!res)
            {
                fprintf(stderr, "Error: allocation failed for output %d\n", i);
                break;
            }
            unflatten_to_image(res, h_output + i * out_stride);
            char *fname = get_filename(image_files[i]);
            char *out_path = join_path(output_path, fname);
            if (out_path)
                write_image(out_path, res);
            free(out_path);
            free(fname);
            free_image(res);
        }

        cudaFree(d_input);
        cudaFree(d_output);
        if (d_kernel) cudaFree(d_kernel);
        free(h_input);
        free(h_output);
        free_images(imgs, num_images);
        free_string_array(image_files, num_images);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else
    {
        Image *img = read_image(input_path);
        if (!img)
        {
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }
        if (img->channels != kernel->input_channels)
        {
            fprintf(stderr, "Error: image has %d channels, kernel expects %d.\n", img->channels, kernel->input_channels);
            free_image(img);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        int img_stride = img->channels * img->height * img->width;
        int out_stride = kernel->output_channels * img->height * img->width;

        unsigned char *h_input = flatten_image_host(img);
        unsigned char *h_output = (unsigned char *)malloc(out_stride);
        if (!h_input || !h_output)
        {
            free(h_input);
            free(h_output);
            free_image(img);
            free(h_kernel);
            free_kernel_set(kset);
            return 1;
        }

        unsigned char *d_input = NULL;
        unsigned char *d_output = NULL;
        float *d_kernel = NULL;
        CUDA_CHECK(cudaMalloc(&d_input, img_stride));
        CUDA_CHECK(cudaMalloc(&d_output, out_stride));
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_elems * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input, img_stride, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_elems * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16, 1);
        dim3 grid((img->width + block.x - 1) / block.x,
                  (img->height + block.y - 1) / block.y,
                  1 * kernel->output_channels);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        convolution_batch_kernel<<<grid, block>>>(d_input, d_output,
                                                  d_kernel,
                                                  img->width, img->height,
                                                  kernel->input_channels, kernel->output_channels,
                                                  kernel->height, kernel->width,
                                                  1, img->max_val);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double seconds = ms / 1000.0;
        printf("Total computation time: %.6f seconds\n", seconds);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_output, d_output, out_stride, cudaMemcpyDeviceToHost));

        Image *res = allocate_result_image(img, kernel);
        if (res)
        {
            unflatten_to_image(res, h_output);
            write_image(output_path, res);
            free_image(res);
        }

        cudaFree(d_input);
        cudaFree(d_output);
        if (d_kernel) cudaFree(d_kernel);
        free(h_input);
        free(h_output);
        free_image(img);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    free(h_kernel);
    free_kernel_set(kset);
    return 0;
}
