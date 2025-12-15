#include "utilities.h"
#include <string.h>

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