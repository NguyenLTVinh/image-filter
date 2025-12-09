#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  int width;
  int height;
  int channels;
  int max_val;
  unsigned char ***data; // [channel][height][width]
} Image;

typedef struct {
  int height;
  int width;
  int input_channels;
  int output_channels;
  float ****data; // [out_ch][in_ch][ky][kx]
} Kernel;

typedef struct {
  int num_kernels;
  Kernel **kernels;
} KernelSet;

void free_image(Image *img);
void free_kernel_set(KernelSet *kset);

void skip_comments(FILE *fp) {
  int c;
  while ((c = fgetc(fp)) == '#') {
    while ((c = fgetc(fp)) != '\n' && c != EOF)
      ;
  }
  ungetc(c, fp);
}

// Read PPM P3 (color) or PGM P2 (grayscale) image
Image *read_image(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return NULL;
  }

  Image *img = (Image *)malloc(sizeof(Image));
  if (!img) {
    fclose(fp);
    return NULL;
  }

  char format[3];
  fscanf(fp, "%2s", format);

  if (strcmp(format, "P3") == 0) {
    img->channels = 3; // RGB
  } else if (strcmp(format, "P2") == 0) {
    img->channels = 1; // Grayscale
  } else {
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
  if (!img->data) {
    free(img);
    fclose(fp);
    return NULL;
  }

  for (int c = 0; c < img->channels; c++) {
    img->data[c] =
        (unsigned char **)malloc(img->height * sizeof(unsigned char *));
    if (!img->data[c]) {
      for (int i = 0; i < c; i++) {
        for (int j = 0; j < img->height; j++) {
          free(img->data[i][j]);
        }
        free(img->data[i]);
      }
      free(img->data);
      free(img);
      fclose(fp);
      return NULL;
    }

    for (int y = 0; y < img->height; y++) {
      img->data[c][y] =
          (unsigned char *)malloc(img->width * sizeof(unsigned char));
      if (!img->data[c][y]) {
        for (int j = 0; j < y; j++) {
          free(img->data[c][j]);
        }
        free(img->data[c]);
        for (int i = 0; i < c; i++) {
          for (int j = 0; j < img->height; j++) {
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
  if (img->channels == 3) {
    // PPM: interleaved RGB
    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
        int r, g, b;
        fscanf(fp, "%d %d %d", &r, &g, &b);
        img->data[0][y][x] = (unsigned char)r;
        img->data[1][y][x] = (unsigned char)g;
        img->data[2][y][x] = (unsigned char)b;
      }
    }
  } else {
    // PGM: single channel
    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
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
int write_image(const char *filename, Image *img) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Error: Cannot create file %s\n", filename);
    return 0;
  }

  if (img->channels == 3) {
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_val);

    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
        fprintf(fp, "%d %d %d\n", img->data[0][y][x], img->data[1][y][x],
                img->data[2][y][x]);
      }
    }
  } else {
    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_val);

    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
        fprintf(fp, "%d\n", img->data[0][y][x]);
      }
    }
  }

  fclose(fp);
  return 1;
}

// Read 3D kernels from text file
KernelSet *read_kernels(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open kernel file %s\n", filename);
    return NULL;
  }

  KernelSet *kset = (KernelSet *)malloc(sizeof(KernelSet));
  if (!kset) {
    fclose(fp);
    return NULL;
  }

  fscanf(fp, "%d", &kset->num_kernels);

  if (kset->num_kernels < 1) {
    fprintf(stderr, "Error: Must have at least 1 kernel\n");
    free(kset);
    fclose(fp);
    return NULL;
  }

  kset->kernels = (Kernel **)calloc(kset->num_kernels, sizeof(Kernel *));
  if (!kset->kernels) {
    free(kset);
    fclose(fp);
    return NULL;
  }

  for (int k = 0; k < kset->num_kernels; k++) {
    kset->kernels[k] = (Kernel *)calloc(1, sizeof(Kernel));
    if (!kset->kernels[k]) {
      free_kernel_set(kset);
      fclose(fp);
      return NULL;
    }

    fscanf(fp, "%d %d %d %d", &kset->kernels[k]->height,
           &kset->kernels[k]->width, &kset->kernels[k]->input_channels,
           &kset->kernels[k]->output_channels);

    if (kset->kernels[k]->height % 2 == 0 || kset->kernels[k]->width % 2 == 0) {
      fprintf(stderr, "Error: Kernel %d dimensions must be odd\n", k + 1);
      free_kernel_set(kset);
      fclose(fp);
      return NULL;
    }

    // Allocate 4D array: [out_ch][in_ch][height][width]
    kset->kernels[k]->data = (float ****)calloc(
        kset->kernels[k]->output_channels, sizeof(float ***));
    if (!kset->kernels[k]->data) {
      free_kernel_set(kset);
      fclose(fp);
      return NULL;
    }

    for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++) {
      kset->kernels[k]->data[oc] =
          (float ***)calloc(kset->kernels[k]->input_channels, sizeof(float **));
      if (!kset->kernels[k]->data[oc]) {
        free_kernel_set(kset);
        fclose(fp);
        return NULL;
      }

      for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++) {
        kset->kernels[k]->data[oc][ic] =
            (float **)calloc(kset->kernels[k]->height, sizeof(float *));
        if (!kset->kernels[k]->data[oc][ic]) {
          free_kernel_set(kset);
          fclose(fp);
          return NULL;
        }

        for (int y = 0; y < kset->kernels[k]->height; y++) {
          kset->kernels[k]->data[oc][ic][y] =
              (float *)malloc(kset->kernels[k]->width * sizeof(float));
          if (!kset->kernels[k]->data[oc][ic][y]) {
            free_kernel_set(kset);
            fclose(fp);
            return NULL;
          }
        }
      }
    }

    // Read kernel values: for each output channel, for each input channel, read
    // the 2D kernel
    for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++) {
      for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++) {
        for (int y = 0; y < kset->kernels[k]->height; y++) {
          for (int x = 0; x < kset->kernels[k]->width; x++) {
            fscanf(fp, "%f", &kset->kernels[k]->data[oc][ic][y][x]);
          }
        }
      }
    }
  }

  fclose(fp);
  return kset;
}

Image *apply_convolution(Image *img, Kernel *kernel, double *time_taken) {
  if (img->channels != kernel->input_channels) {
    fprintf(stderr, "Error: Image has %d channels but kernel expects %d\n",
            img->channels, kernel->input_channels);
    return NULL;
  }

  Image *result = (Image *)calloc(1, sizeof(Image));
  if (!result)
    return NULL;

  result->width = img->width;
  result->height = img->height;
  result->channels = kernel->output_channels;
  result->max_val = img->max_val;

  result->data =
      (unsigned char ***)calloc(result->channels, sizeof(unsigned char **));
  if (!result->data) {
    free(result);
    return NULL;
  }

  for (int c = 0; c < result->channels; c++) {
    result->data[c] =
        (unsigned char **)calloc(result->height, sizeof(unsigned char *));
    if (!result->data[c]) {
      free_image(result);
      return NULL;
    }

    for (int y = 0; y < result->height; y++) {
      result->data[c][y] =
          (unsigned char *)malloc(result->width * sizeof(unsigned char));
      if (!result->data[c][y]) {
        free_image(result);
        return NULL;
      }
    }
  }

  int offset_y = kernel->height / 2;
  int offset_x = kernel->width / 2;

  clock_t start = clock();

  // For each output channel
  for (int oc = 0; oc < kernel->output_channels; oc++) {
    // For each output pixel
    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
        float sum = 0.0;

        // Convolve across all input channels
        for (int ic = 0; ic < kernel->input_channels; ic++) {
          for (int ky = 0; ky < kernel->height; ky++) {
            for (int kx = 0; kx < kernel->width; kx++) {
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

void free_image(Image *img) {
  if (img) {
    if (img->data) {
      for (int c = 0; c < img->channels; c++) {
        if (img->data[c]) {
          for (int y = 0; y < img->height; y++) {
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

void free_kernel_set(KernelSet *kset) {
  if (kset) {
    if (kset->kernels) {
      for (int k = 0; k < kset->num_kernels; k++) {
        if (kset->kernels[k]) {
          if (kset->kernels[k]->data) {
            for (int oc = 0; oc < kset->kernels[k]->output_channels; oc++) {
              if (kset->kernels[k]->data[oc]) {
                for (int ic = 0; ic < kset->kernels[k]->input_channels; ic++) {
                  if (kset->kernels[k]->data[oc][ic]) {
                    for (int y = 0; y < kset->kernels[k]->height; y++) {
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

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <input.ppm|input.pgm> <kernel.txt> "
           "<output.ppm|output.pgm>\n",
           argv[0]);
    printf("\nSupported formats:\n");
    printf("  Input: P3 (PPM color) or P2 (PGM grayscale)\n");
    printf("  Output: Determined by number of output channels\n");
    printf("\nKernel file format:\n");
    printf("  Line 1: <num_kernels>\n");
    printf("  For each kernel:\n");
    printf("    Line 1: <height> <width> <input_channels> <output_channels>\n");
    printf("    Following lines: kernel values\n");
    printf(
        "      (for each output ch, for each input ch, height×width values)\n");
    return 1;
  }

  Image *img = read_image(argv[1]);
  if (!img) {
    return 1;
  }

  KernelSet *kset = read_kernels(argv[2]);
  if (!kset) {
    free_image(img);
    return 1;
  }

  printf("Processing image: %dx%dx%d with %d kernel(s)\n", img->width,
         img->height, img->channels, kset->num_kernels);

  Image *current = img;
  double total_time = 0.0;

  for (int k = 0; k < kset->num_kernels; k++) {
    printf("Applying kernel %d/%d (%dx%d, %d->%d channels)...\n", k + 1,
           kset->num_kernels, kset->kernels[k]->height, kset->kernels[k]->width,
           kset->kernels[k]->input_channels, kset->kernels[k]->output_channels);

    double kernel_time = 0.0;
    Image *result = apply_convolution(current, kset->kernels[k], &kernel_time);

    if (!result) {
      fprintf(stderr, "Error: Convolution failed on kernel %d\n", k + 1);
      if (current != img)
        free_image(current);
      free_image(img);
      free_kernel_set(kset);
      return 1;
    }

    printf("  Computation time: %.6f seconds\n", kernel_time);
    total_time += kernel_time;

    if (current != img) {
      free_image(current);
    }
    current = result;
  }

  printf("\nTotal computation time: %.6f seconds\n", total_time);
  printf("Output: %dx%dx%d\n", current->width, current->height,
         current->channels);

  if (write_image(argv[3], current)) {
    printf("Output written to %s\n", argv[3]);
  }

  free_image(img);
  if (current != img)
    free_image(current);
  free_kernel_set(kset);

  return 0;
}
