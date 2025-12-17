#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "utilities.h"
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>

void skip_comments(FILE *fp)
{
    int c;
    while ((c = fgetc(fp)) == '#')
    {
        while ((c = fgetc(fp)) != '\n' && c != EOF)
            ;
    }
    ungetc(c, fp);
}

// Read PPM P3 (color) or PGM P2 (grayscale) image
Image *read_image(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    Image *img = (Image *)malloc(sizeof(Image));
    if (!img)
    {
        fclose(fp);
        return NULL;
    }

    char format[3];
    fscanf(fp, "%2s", format);

    if (strcmp(format, "P3") == 0)
    {
        img->channels = 3; // RGB
    }
    else if (strcmp(format, "P2") == 0)
    {
        img->channels = 1; // Grayscale
    }
    else
    {
        fprintf(stderr, "Error: Only P3 (PPM) and P2 (PGM) formats supported\n");
        free(img);
        fclose(fp);
        return NULL;
    }

    skip_comments(fp);
    fscanf(fp, "%d %d", &img->width, &img->height);
    skip_comments(fp);
    fscanf(fp, "%d", &img->max_val);

    // Allocate 3D array: [channels][height][width]
    img->data =
        (unsigned char ***)malloc(img->channels * sizeof(unsigned char **));
    if (!img->data)
    {
        free(img);
        fclose(fp);
        return NULL;
    }

    for (int c = 0; c < img->channels; c++)
    {
        img->data[c] =
            (unsigned char **)malloc(img->height * sizeof(unsigned char *));
        if (!img->data[c])
        {
            for (int i = 0; i < c; i++)
            {
                for (int j = 0; j < img->height; j++)
                {
                    free(img->data[i][j]);
                }
                free(img->data[i]);
            }
            free(img->data);
            free(img);
            fclose(fp);
            return NULL;
        }

        for (int y = 0; y < img->height; y++)
        {
            img->data[c][y] =
                (unsigned char *)malloc(img->width * sizeof(unsigned char));
            if (!img->data[c][y])
            {
                for (int j = 0; j < y; j++)
                {
                    free(img->data[c][j]);
                }
                free(img->data[c]);
                for (int i = 0; i < c; i++)
                {
                    for (int j = 0; j < img->height; j++)
                    {
                        free(img->data[i][j]);
                    }
                    free(img->data[i]);
                }
                free(img->data);
                free(img);
                fclose(fp);
                return NULL;
            }
        }
    }

    // Read pixel data
    if (img->channels == 3)
    {
        // PPM: interleaved RGB
        for (int y = 0; y < img->height; y++)
        {
            for (int x = 0; x < img->width; x++)
            {
                int r, g, b;
                fscanf(fp, "%d %d %d", &r, &g, &b);
                img->data[0][y][x] = (unsigned char)r;
                img->data[1][y][x] = (unsigned char)g;
                img->data[2][y][x] = (unsigned char)b;
            }
        }
    }
    else
    {
        // PGM: single channel
        for (int y = 0; y < img->height; y++)
        {
            for (int x = 0; x < img->width; x++)
            {
                int val;
                fscanf(fp, "%d", &val);
                img->data[0][y][x] = (unsigned char)val;
            }
        }
    }

    fclose(fp);
    return img;
}

// Write image (PPM P3 for color, PGM P2 for grayscale)
int write_image(const char *filename, Image *img)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return 0;
    }

    if (img->channels == 3)
    {
        fprintf(fp, "P3\n");
        fprintf(fp, "%d %d\n", img->width, img->height);
        fprintf(fp, "%d\n", img->max_val);

        for (int y = 0; y < img->height; y++)
        {
            for (int x = 0; x < img->width; x++)
            {
                fprintf(fp, "%d %d %d\n", img->data[0][y][x], img->data[1][y][x],
                        img->data[2][y][x]);
            }
        }
    }
    else
    {
        fprintf(fp, "P2\n");
        fprintf(fp, "%d %d\n", img->width, img->height);
        fprintf(fp, "%d\n", img->max_val);

        for (int y = 0; y < img->height; y++)
        {
            for (int x = 0; x < img->width; x++)
            {
                fprintf(fp, "%d\n", img->data[0][y][x]);
            }
        }
    }

    fclose(fp);
    return 1;
}

// Read 3D kernels from text file
KernelSet *read_kernels(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open kernel file %s\n", filename);
        return NULL;
    }

    KernelSet *kset = (KernelSet *)malloc(sizeof(KernelSet));
    if (!kset)
    {
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d", &kset->num_kernels);

    if (kset->num_kernels < 1)
    {
        fprintf(stderr, "Error: Must have at least 1 kernel\n");
        free(kset);
        fclose(fp);
        return NULL;
    }

    kset->kernels = (Kernel **)calloc(kset->num_kernels, sizeof(Kernel *));
    if (!kset->kernels)
    {
        free(kset);
        fclose(fp);
        return NULL;
    }

    for (int k = 0; k < kset->num_kernels; k++)
    {
        kset->kernels[k] = (Kernel *)calloc(1, sizeof(Kernel));
        if (!kset->kernels[k])
        {
            free_kernel_set(kset);
            fclose(fp);
            return NULL;
        }

        fscanf(fp, "%d %d %d %d", &kset->kernels[k]->height,
               &kset->kernels[k]->width, &kset->kernels[k]->input_channels,
               &kset->kernels[k]->output_channels);

        if (kset->kernels[k]->height % 2 == 0 || kset->kernels[k]->width % 2 == 0)
        {
            fprintf(stderr, "Error: Kernel %d dimensions must be odd\n", k + 1);
            free_kernel_set(kset);
            fclose(fp);
            return NULL;
        }

        // Allocate 4D array: [out_ch][in_ch][height][width]
        kset->kernels[k]->data = (float ****)calloc(
            kset->kernels[k]->output_channels, sizeof(float ***));
        if (!kset->kernels[k]->data)
        {
            free_kernel_set(kset);
            fclose(fp);
            return NULL;
        }

        for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++)
        {
            kset->kernels[k]->data[oc] =
                (float ***)calloc(kset->kernels[k]->input_channels, sizeof(float **));
            if (!kset->kernels[k]->data[oc])
            {
                free_kernel_set(kset);
                fclose(fp);
                return NULL;
            }

            for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++)
            {
                kset->kernels[k]->data[oc][ic] =
                    (float **)calloc(kset->kernels[k]->height, sizeof(float *));
                if (!kset->kernels[k]->data[oc][ic])
                {
                    free_kernel_set(kset);
                    fclose(fp);
                    return NULL;
                }

                for (int y = 0; y < kset->kernels[k]->height; y++)
                {
                    kset->kernels[k]->data[oc][ic][y] =
                        (float *)malloc(kset->kernels[k]->width * sizeof(float));
                    if (!kset->kernels[k]->data[oc][ic][y])
                    {
                        free_kernel_set(kset);
                        fclose(fp);
                        return NULL;
                    }
                }
            }
        }

        // Read kernel values: for each output channel, for each input channel, read
        // the 2D kernel
        for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++)
        {
            for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++)
            {
                for (int y = 0; y < kset->kernels[k]->height; y++)
                {
                    for (int x = 0; x < kset->kernels[k]->width; x++)
                    {
                        fscanf(fp, "%f", &kset->kernels[k]->data[oc][ic][y][x]);
                    }
                }
            }
        }
    }

    fclose(fp);
    return kset;
}

void free_image(Image *img)
{
    if (img)
    {
        if (img->data)
        {
            for (int c = 0; c < img->channels; c++)
            {
                if (img->data[c])
                {
                    for (int y = 0; y < img->height; y++)
                    {
                        free(img->data[c][y]);
                    }
                    free(img->data[c]);
                }
            }
            free(img->data);
        }
        free(img);
    }
}

void free_kernel_set(KernelSet *kset)
{
    if (kset)
    {
        if (kset->kernels)
        {
            for (int k = 0; k < kset->num_kernels; k++)
            {
                if (kset->kernels[k])
                {
                    if (kset->kernels[k]->data)
                    {
                        for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++)
                        {
                            if (kset->kernels[k]->data[oc])
                            {
                                for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++)
                                {
                                    if (kset->kernels[k]->data[oc][ic])
                                    {
                                        for (int y = 0; y < kset->kernels[k]->height; y++)
                                        {
                                            free(kset->kernels[k]->data[oc][ic][y]);
                                        }
                                        free(kset->kernels[k]->data[oc][ic]);
                                    }
                                }
                                free(kset->kernels[k]->data[oc]);
                            }
                        }
                        free(kset->kernels[k]->data);
                    }
                    free(kset->kernels[k]);
                }
            }
            free(kset->kernels);
        }
        free(kset);
    }
}

int is_image_file(const char *filename)
{
    const char *dot = strrchr(filename, '.');
    if (!dot)
        return 0;
    return (strcmp(dot, ".ppm") == 0 || strcmp(dot, ".pgm") == 0);
}

char **get_image_files(const char *directory, int *count)
{
    DIR *dir = opendir(directory);
    if (!dir)
    {
        fprintf(stderr, "Error: Cannot open directory %s\n", directory);
        *count = 0;
        return NULL;
    }

    // First pass: count image files
    *count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG && is_image_file(entry->d_name))
        {
            (*count)++;
        }
    }

    if (*count == 0)
    {
        closedir(dir);
        return NULL;
    }

    // Allocate array for filenames
    char **files = (char **)malloc(*count * sizeof(char *));
    if (!files)
    {
        closedir(dir);
        *count = 0;
        return NULL;
    }

    // Second pass: store filenames
    rewinddir(dir);
    int idx = 0;
    while ((entry = readdir(dir)) != NULL && idx < *count)
    {
        if (entry->d_type == DT_REG && is_image_file(entry->d_name))
        {
            size_t len = strlen(directory) + strlen(entry->d_name) + 2;
            files[idx] = (char *)malloc(len);
            if (!files[idx])
            {
                free_string_array(files, idx);
                closedir(dir);
                *count = 0;
                return NULL;
            }
            snprintf(files[idx], len, "%s/%s", directory, entry->d_name);
            idx++;
        }
    }

    closedir(dir);
    return files;
}

void free_string_array(char **array, int count)
{
    if (array)
    {
        for (int i = 0; i < count; i++)
        {
            free(array[i]);
        }
        free(array);
    }
}

int create_directory(const char *path)
{
    struct stat st = {0};
    if (stat(path, &st) == -1)
    {
        if (mkdir(path, 0755) != 0)
        {
            fprintf(stderr, "Error: Cannot create directory %s\n", path);
            return 0;
        }
    }
    return 1;
}

char *get_filename(const char *path)
{
    const char *filename = strrchr(path, '/');
    if (filename)
        return strdup(filename + 1);
    return strdup(path);
}

char *join_path(const char *dir, const char *filename)
{
    size_t len = strlen(dir) + strlen(filename) + 2;
    char *path = (char *)malloc(len);
    if (path)
    {
        snprintf(path, len, "%s/%s", dir, filename);
    }
    return path;
}

int is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0)
    {
        return S_ISDIR(st.st_mode);
    }
    return 0;
}

unsigned char *flatten_image_host(const Image *img)
{
    int size = img->channels * img->height * img->width;
    unsigned char *flat = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (!flat)
        return NULL;

    for (int c = 0; c < img->channels; c++)
    {
        for (int y = 0; y < img->height; y++)
        {
            memcpy(flat + c * img->height * img->width + y * img->width, img->data[c][y], img->width);
        }
    }
    return flat;
}

float *flatten_kernel_host(const Kernel *kernel)
{
    int kernel_size = kernel->output_channels * kernel->input_channels * kernel->height * kernel->width;
    float *flat = (float *)malloc(kernel_size * sizeof(float));
    if (!flat)
        return NULL;

    int idx = 0;
    for (int oc = 0; oc < kernel->output_channels; oc++)
    {
        for (int ic = 0; ic < kernel->input_channels; ic++)
        {
            for (int ky = 0; ky < kernel->height; ky++)
            {
                for (int kx = 0; kx < kernel->width; kx++)
                {
                    flat[idx++] = kernel->data[oc][ic][ky][kx];
                }
            }
        }
    }
    return flat;
}

Image *allocate_result_image(const Image *img, const Kernel *kernel)
{
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

    return result;
}

void unflatten_to_image(Image *result, const unsigned char *flat)
{
    int stride = result->height * result->width;
    for (int c = 0; c < result->channels; c++)
    {
        const unsigned char *src = flat + c * stride;
        for (int y = 0; y < result->height; y++)
        {
            memcpy(result->data[c][y], src + y * result->width, result->width);
        }
    }
}

int run_filter(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <input> <kernel.txt> <output>\n", argv[0]);
        printf("\n  input:  Single image file (.ppm/.pgm) OR directory of images\n");
        printf("  output: Single output file OR output directory\n");
        printf("\nFor directory mode: processes images in batches of 256\n");
        printf("\nSupported formats:\n");
        printf("  Input: P3 (PPM color) or P2 (PGM grayscale)\n");
        printf("  Output: Determined by number of output channels\n");
        printf("\nKernel file format:\n");
        printf("  Line 1: <num_kernels>\n");
        printf("  For each kernel:\n");
        printf("    Line 1: <height> <width> <input_channels> <output_channels>\n");
        printf("    Following lines: kernel values\n");
        printf("      (for each output ch, for each input ch, height×width values)\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *kernel_path = argv[2];
    const char *output_path = argv[3];

    if (is_directory(input_path))
    {
        if (!create_directory(output_path))
        {
            return 1;
        }

        int num_images = 0;
        char **image_files = get_image_files(input_path, &num_images);
        if (!image_files || num_images == 0)
        {
            fprintf(stderr, "Error: No image files found in %s\n", input_path);
            return 1;
        }

        KernelSet *kset = read_kernels(kernel_path);
        if (!kset)
        {
            free_string_array(image_files, num_images);
            return 1;
        }

        const int BATCH_SIZE = 256;
        double total_time = 0.0;
        int processed = 0;

        for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > num_images)
                batch_end = num_images;

            printf("\nBatch %d-%d:\n", batch_start + 1, batch_end);

            for (int i = batch_start; i < batch_end; i++)
            {
                char *filename = get_filename(image_files[i]);
                printf("  [%d/%d] %s... ", i + 1, num_images, filename);
                fflush(stdout);

                Image *img = read_image(image_files[i]);
                if (!img)
                {
                    printf("FAILED (read)\n");
                    free(filename);
                    continue;
                }

                Image *current = img;
                int success = 1;

                for (int k = 0; k < kset->num_kernels; k++)
                {
                    double kernel_time = 0.0;
                    Image *result = apply_convolution(current, kset->kernels[k], &kernel_time);

                    if (!result)
                    {
                        if (current != img)
                            free_image(current);
                        free_image(img);
                        printf("FAILED (kernel %d)\n", k + 1);
                        success = 0;
                        break;
                    }

                    total_time += kernel_time;

                    if (current != img)
                        free_image(current);
                    current = result;
                }

                if (success)
                {
                    char *output_file = join_path(output_path, filename);
                    if (write_image(output_file, current))
                    {
                        printf("OK\n");
                        processed++;
                    }
                    else
                    {
                        printf("FAILED (write)\n");
                    }
                    free(output_file);
                }

                free_image(img);
                if (current != img)
                    free_image(current);
                free(filename);
            }
        }

        printf("\n=== Summary ===\n");
        printf("Processed: %d/%d images\n", processed, num_images);
        printf("Total computation time: %.6f seconds\n", total_time);

        free_kernel_set(kset);
        free_string_array(image_files, num_images);

        return processed < num_images ? 1 : 0;
    }
    else
    {
        Image *img = read_image(input_path);
        if (!img)
            return 1;

        KernelSet *kset = read_kernels(kernel_path);
        if (!kset)
        {
            free_image(img);
            return 1;
        }

        printf("Processing image: %dx%dx%d with %d kernel(s)\n", img->width, img->height, img->channels, kset->num_kernels);

        Image *current = img;
        double total_time = 0.0;

        for (int k = 0; k < kset->num_kernels; k++)
        {
            printf("Applying kernel %d/%d (%dx%d, %d->%d channels)...\n", k + 1, kset->num_kernels, kset->kernels[k]->height, kset->kernels[k]->width, kset->kernels[k]->input_channels, kset->kernels[k]->output_channels);

            double kernel_time = 0.0;
            Image *result = apply_convolution(current, kset->kernels[k], &kernel_time);

            if (!result)
            {
                fprintf(stderr, "Error: Convolution failed on kernel %d\n", k + 1);
                if (current != img)
                    free_image(current);
                free_image(img);
                free_kernel_set(kset);
                return 1;
            }

            printf("  Computation time: %.6f seconds\n", kernel_time);
            total_time += kernel_time;

            if (current != img)
                free_image(current);
            current = result;
        }

        printf("\nTotal computation time: %.6f seconds\n", total_time);
        printf("Output: %dx%dx%d\n", current->width, current->height, current->channels);

        if (write_image(output_path, current))
            printf("Output written to %s\n", output_path);

        free_image(img);
        if (current != img)
            free_image(current);
        free_kernel_set(kset);

        return 0;
    }
}
