#include "../utilities.h"
#include <time.h>

Image *apply_convolution(Image *img, Kernel *kernel, double *time_taken)
{
    if (img->channels != kernel->input_channels)
    {
        fprintf(stderr, "Error: Image has %d channels but kernel expects %d\n", img->channels, kernel->input_channels);
        return NULL;
    }

    Image *result = (Image *)calloc(1, sizeof(Image));
    if (!result)
        return NULL;

    result->width = img->width;
    result->height = img->height;
    result->channels = kernel->output_channels;
    result->max_val = img->max_val;

    result->data = (unsigned char ***)calloc(result->channels, sizeof(unsigned char **));
    if (!result->data)
    {
        free(result);
        return NULL;
    }

    for (int c = 0; c < result->channels; c++)
    {
        result->data[c] = (unsigned char **)calloc(result->height, sizeof(unsigned char *));
        if (!result->data[c])
        {
            free_image(result);
            return NULL;
        }

        for (int y = 0; y < result->height; y++)
        {
            result->data[c][y] = (unsigned char *)malloc(result->width * sizeof(unsigned char));
            if (!result->data[c][y])
            {
                free_image(result);
                return NULL;
            }
        }
    }

    int offset_y = kernel->height / 2;
    int offset_x = kernel->width / 2;

    clock_t start = clock();

    for (int oc = 0; oc < kernel->output_channels; oc++)
    {
        for (int y = 0; y < img->height; y++)
        {
            for (int x = 0; x < img->width; x++)
            {
                float sum = 0.0f;

                for (int ic = 0; ic < kernel->input_channels; ic++)
                {
                    for (int ky = 0; ky < kernel->height; ky++)
                    {
                        for (int kx = 0; kx < kernel->width; kx++)
                        {
                            int py = y + ky - offset_y;
                            int px = x + kx - offset_x;

                            if (py < 0)
                                py = 0;
                            if (py >= img->height)
                                py = img->height - 1;
                            if (px < 0)
                                px = 0;
                            if (px >= img->width)
                                px = img->width - 1;

                            float pixel_val = (float)img->data[ic][py][px];
                            float kernel_val = kernel->data[oc][ic][ky][kx];
                            sum += pixel_val * kernel_val;
                        }
                    }
                }

                if (sum < 0)
                    sum = 0;
                if (sum > img->max_val)
                    sum = img->max_val;

                result->data[oc][y][x] = (unsigned char)sum;
            }
        }
    }

    clock_t end = clock();
    *time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    return result;
}

int main(int argc, char *argv[])
{
    return run_filter(argc, argv);
}
