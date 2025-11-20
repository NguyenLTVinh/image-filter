#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  int width;
  int height;
  int max_val;
  unsigned char *r;
  unsigned char *g;
  unsigned char *b;
} PPMImage;

typedef struct {
  int size;
  float **data;
} Kernel;

typedef struct {
  int num_kernels;
  Kernel **kernels;
} KernelSet;

void skip_comments(FILE *fp) {
  int c;
  while ((c = fgetc(fp)) == '#') {
    while ((c = fgetc(fp)) != '\n' && c != EOF)
      ;
  }
  ungetc(c, fp);
}

// Read PPM P3 image
PPMImage *read_ppm(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return NULL;
  }

  PPMImage *img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fclose(fp);
    return NULL;
  }

  char format[3];
  fscanf(fp, "%2s", format);

  if (strcmp(format, "P3") != 0) {
    fprintf(stderr, "Error: Only P3 PPM format is supported\n");
    free(img);
    fclose(fp);
    return NULL;
  }

  skip_comments(fp);
  fscanf(fp, "%d %d", &img->width, &img->height);
  skip_comments(fp);
  fscanf(fp, "%d", &img->max_val);

  int size = img->width * img->height;
  img->r = (unsigned char *)malloc(size);
  img->g = (unsigned char *)malloc(size);
  img->b = (unsigned char *)malloc(size);

  if (!img->r || !img->g || !img->b) {
    free(img->r);
    free(img->g);
    free(img->b);
    free(img);
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < size; i++) {
    int r, g, b;
    fscanf(fp, "%d %d %d", &r, &g, &b);
    img->r[i] = (unsigned char)r;
    img->g[i] = (unsigned char)g;
    img->b[i] = (unsigned char)b;
  }

  fclose(fp);
  return img;
}

// Write PPM P3 image
int write_ppm(const char *filename, PPMImage *img) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Error: Cannot create file %s\n", filename);
    return 0;
  }

  fprintf(fp, "P3\n");
  fprintf(fp, "%d %d\n", img->width, img->height);
  fprintf(fp, "%d\n", img->max_val);

  for (int i = 0; i < img->width * img->height; i++) {
    fprintf(fp, "%d %d %d\n", img->r[i], img->g[i], img->b[i]);
  }

  fclose(fp);
  return 1;
}

// Read kernels from text file (supports multiple kernels)
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

  // Read number of kernels
  fscanf(fp, "%d", &kset->num_kernels);

  if (kset->num_kernels < 1) {
    fprintf(stderr, "Error: Must have at least 1 kernel\n");
    free(kset);
    fclose(fp);
    return NULL;
  }

  kset->kernels = (Kernel **)malloc(kset->num_kernels * sizeof(Kernel *));
  if (!kset->kernels) {
    free(kset);
    fclose(fp);
    return NULL;
  }

  // Read each kernel
  for (int k = 0; k < kset->num_kernels; k++) {
    kset->kernels[k] = (Kernel *)malloc(sizeof(Kernel));
    if (!kset->kernels[k]) {
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < kset->kernels[i]->size; j++) {
          free(kset->kernels[i]->data[j]);
        }
        free(kset->kernels[i]->data);
        free(kset->kernels[i]);
      }
      free(kset->kernels);
      free(kset);
      fclose(fp);
      return NULL;
    }

    fscanf(fp, "%d", &kset->kernels[k]->size);

    if (kset->kernels[k]->size % 2 == 0) {
      fprintf(stderr, "Error: Kernel %d size must be odd\n", k + 1);
      for (int i = 0; i <= k; i++) {
        if (i < k) {
          for (int j = 0; j < kset->kernels[i]->size; j++) {
            free(kset->kernels[i]->data[j]);
          }
          free(kset->kernels[i]->data);
        }
        free(kset->kernels[i]);
      }
      free(kset->kernels);
      free(kset);
      fclose(fp);
      return NULL;
    }

    kset->kernels[k]->data =
        (float **)malloc(kset->kernels[k]->size * sizeof(float *));
    if (!kset->kernels[k]->data) {
      for (int i = 0; i <= k; i++) {
        if (i < k) {
          for (int j = 0; j < kset->kernels[i]->size; j++) {
            free(kset->kernels[i]->data[j]);
          }
          free(kset->kernels[i]->data);
        }
        free(kset->kernels[i]);
      }
      free(kset->kernels);
      free(kset);
      fclose(fp);
      return NULL;
    }

    for (int i = 0; i < kset->kernels[k]->size; i++) {
      kset->kernels[k]->data[i] =
          (float *)malloc(kset->kernels[k]->size * sizeof(float));
      if (!kset->kernels[k]->data[i]) {
        for (int j = 0; j < i; j++) {
          free(kset->kernels[k]->data[j]);
        }
        free(kset->kernels[k]->data);
        for (int j = 0; j < k; j++) {
          for (int m = 0; m < kset->kernels[j]->size; m++) {
            free(kset->kernels[j]->data[m]);
          }
          free(kset->kernels[j]->data);
          free(kset->kernels[j]);
        }
        free(kset->kernels[k]);
        free(kset->kernels);
        free(kset);
        fclose(fp);
        return NULL;
      }
      for (int j = 0; j < kset->kernels[k]->size; j++) {
        fscanf(fp, "%f", &kset->kernels[k]->data[i][j]);
      }
    }
  }

  fclose(fp);
  return kset;
}

PPMImage *apply_convolution(PPMImage *img, Kernel *kernel, double *time_taken) {
  PPMImage *result = (PPMImage *)malloc(sizeof(PPMImage));
  if (!result)
    return NULL;

  result->width = img->width;
  result->height = img->height;
  result->max_val = img->max_val;

  int size = img->width * img->height;
  result->r = (unsigned char *)malloc(size);
  result->g = (unsigned char *)malloc(size);
  result->b = (unsigned char *)malloc(size);

  if (!result->r || !result->g || !result->b) {
    free(result->r);
    free(result->g);
    free(result->b);
    free(result);
    return NULL;
  }

  int offset = kernel->size / 2;

  // Start timing here - just before first pixel calculation
  clock_t start = clock();

  for (int y = 0; y < img->height; y++) {
    for (int x = 0; x < img->width; x++) {
      float sum_r = 0, sum_g = 0, sum_b = 0;

      for (int ky = 0; ky < kernel->size; ky++) {
        for (int kx = 0; kx < kernel->size; kx++) {
          int py = y + ky - offset;
          int px = x + kx - offset;

          if (py < 0)
            py = 0;
          if (py >= img->height)
            py = img->height - 1;
          if (px < 0)
            px = 0;
          if (px >= img->width)
            px = img->width - 1;

          int idx = py * img->width + px;
          float k = kernel->data[ky][kx];

          sum_r += img->r[idx] * k;
          sum_g += img->g[idx] * k;
          sum_b += img->b[idx] * k;
        }
      }

      if (sum_r < 0)
        sum_r = 0;
      if (sum_r > img->max_val)
        sum_r = img->max_val;
      if (sum_g < 0)
        sum_g = 0;
      if (sum_g > img->max_val)
        sum_g = img->max_val;
      if (sum_b < 0)
        sum_b = 0;
      if (sum_b > img->max_val)
        sum_b = img->max_val;

      int idx = y * img->width + x;
      result->r[idx] = (unsigned char)sum_r;
      result->g[idx] = (unsigned char)sum_g;
      result->b[idx] = (unsigned char)sum_b;
    }
  }

  // End timing after last pixel
  clock_t end = clock();
  *time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

  return result;
}

void free_image(PPMImage *img) {
  if (img) {
    free(img->r);
    free(img->g);
    free(img->b);
    free(img);
  }
}

void free_kernel_set(KernelSet *kset) {
  if (kset) {
    for (int k = 0; k < kset->num_kernels; k++) {
      if (kset->kernels[k]) {
        for (int i = 0; i < kset->kernels[k]->size; i++) {
          free(kset->kernels[k]->data[i]);
        }
        free(kset->kernels[k]->data);
        free(kset->kernels[k]);
      }
    }
    free(kset->kernels);
    free(kset);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <input.ppm> <kernel.txt> <output.ppm>\n", argv[0]);
    printf("\nKernel file format:\n");
    printf("  First line: number of kernels\n");
    printf("  For each kernel:\n");
    printf("    - kernel size (odd number)\n");
    printf("    - kernel values (space-separated)\n");
    return 1;
  }

  PPMImage *img = read_ppm(argv[1]);
  if (!img) {
    return 1;
  }

  KernelSet *kset = read_kernels(argv[2]);
  if (!kset) {
    free_image(img);
    return 1;
  }

  printf("Processing image: %dx%d with %d kernel(s)\n", img->width, img->height,
         kset->num_kernels);

  PPMImage *current = img;
  double total_time = 0.0;

  for (int k = 0; k < kset->num_kernels; k++) {
    printf("Applying kernel %d/%d (%dx%d)...\n", k + 1, kset->num_kernels,
           kset->kernels[k]->size, kset->kernels[k]->size);

    double kernel_time = 0.0;
    PPMImage *result =
        apply_convolution(current, kset->kernels[k], &kernel_time);

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

    // Free intermediate result (except the original image)
    if (current != img) {
      free_image(current);
    }
    current = result;
  }

  printf("\nTotal computation time: %.6f seconds\n", total_time);

  if (write_ppm(argv[3], current)) {
    printf("Output written to %s\n", argv[3]);
  }

  free_image(img);
  if (current != img)
    free_image(current);
  free_kernel_set(kset);

  return 0;
}