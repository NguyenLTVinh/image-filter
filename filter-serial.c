#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <input> <kernel.txt> <output>\n", argv[0]);
    printf(
        "\n  input:  Single image file (.ppm/.pgm) OR directory of images\n");
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
    printf(
        "      (for each output ch, for each input ch, height×width values)\n");
    return 1;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[3];

  // Check if input is a directory or single file
  if (is_directory(input_path)) {
    // Directory mode: batch processing
    if (!create_directory(output_path)) {
      return 1;
    }

    int num_images = 0;
    char **image_files = get_image_files(input_path, &num_images);
    if (!image_files || num_images == 0) {
      fprintf(stderr, "Error: No image files found in %s\n", input_path);
      return 1;
    }

    printf("Found %d image(s) to process\n", num_images);

    KernelSet *kset = read_kernels(argv[2]);
    if (!kset) {
      free_string_array(image_files, num_images);
      return 1;
    }

    const int BATCH_SIZE = 256;
    double total_time = 0.0;
    int processed = 0;

    for (int batch_start = 0; batch_start < num_images;
         batch_start += BATCH_SIZE) {
      int batch_end = batch_start + BATCH_SIZE;
      if (batch_end > num_images)
        batch_end = num_images;

      printf("\nBatch %d-%d:\n", batch_start + 1, batch_end);

      for (int i = batch_start; i < batch_end; i++) {
        char *filename = get_filename(image_files[i]);
        printf("  [%d/%d] %s... ", i + 1, num_images, filename);
        fflush(stdout);

        Image *img = read_image(image_files[i]);
        if (!img) {
          printf("FAILED (read)\n");
          free(filename);
          continue;
        }

        Image *current = img;
        for (int k = 0; k < kset->num_kernels; k++) {
          double kernel_time = 0.0;
          Image *result =
              apply_convolution(current, kset->kernels[k], &kernel_time);

          if (!result) {
            if (current != img)
              free_image(current);
            free_image(img);
            printf("FAILED (kernel %d)\n", k + 1);
            free(filename);
            goto next_image;
          }

          total_time += kernel_time;

          if (current != img) {
            free_image(current);
          }
          current = result;
        }

        char *output_file = join_path(output_path, filename);
        if (write_image(output_file, current)) {
          printf("OK\n");
          processed++;
        } else {
          printf("FAILED (write)\n");
        }

        free_image(img);
        if (current != img)
          free_image(current);
        free(output_file);
        free(filename);

      next_image:
        continue;
      }
    }

    printf("\n=== Summary ===\n");
    printf("Processed: %d/%d images\n", processed, num_images);
    printf("Total computation time: %.6f seconds\n", total_time);

    free_kernel_set(kset);
    free_string_array(image_files, num_images);

    return processed < num_images ? 1 : 0;
  } else {
    // Single image mode: original behavior
    Image *img = read_image(input_path);
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
             kset->num_kernels, kset->kernels[k]->height,
             kset->kernels[k]->width, kset->kernels[k]->input_channels,
             kset->kernels[k]->output_channels);

      double kernel_time = 0.0;
      Image *result =
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

      if (current != img) {
        free_image(current);
      }
      current = result;
    }

    printf("\nTotal computation time: %.6f seconds\n", total_time);
    printf("Output: %dx%dx%d\n", current->width, current->height,
           current->channels);

    if (write_image(output_path, current)) {
      printf("Output written to %s\n", output_path);
    }

    free_image(img);
    if (current != img)
      free_image(current);
    free_kernel_set(kset);

    return 0;
  }
}
