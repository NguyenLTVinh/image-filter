// Optimized batch processing and high-performance convolution
#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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

  double start = omp_get_wtime();

  // Parallelize over output channels and rows
  #pragma omp parallel for collapse(2) schedule(guided)
  for (int oc = 0; oc < kernel->output_channels; oc++) {
    for (int y = 0; y < img->height; y++) {
      
      // Process multiple pixels per iteration
      int x;
      for (x = 0; x <= img->width - 4; x += 4) {
        float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        
        // Unrolled convolution for 4 adjacent pixels
        for (int ic = 0; ic < kernel->input_channels; ic++) {
          for (int ky = 0; ky < kernel->height; ky++) {
            int py = y + ky - offset_y;
            if (py < 0) py = 0;
            else if (py >= img->height) py = img->height - 1;
            
            for (int kx = 0; kx < kernel->width; kx++) {
              float kval = kernel->data[oc][ic][ky][kx];
              
              int px0 = x + kx - offset_x;
              int px1 = x + 1 + kx - offset_x;
              int px2 = x + 2 + kx - offset_x;
              int px3 = x + 3 + kx - offset_x;
              
              if (px0 < 0) px0 = 0; else if (px0 >= img->width) px0 = img->width - 1;
              if (px1 < 0) px1 = 0; else if (px1 >= img->width) px1 = img->width - 1;
              if (px2 < 0) px2 = 0; else if (px2 >= img->width) px2 = img->width - 1;
              if (px3 < 0) px3 = 0; else if (px3 >= img->width) px3 = img->width - 1;
              
              sum0 += (float)img->data[ic][py][px0] * kval;
              sum1 += (float)img->data[ic][py][px1] * kval;
              sum2 += (float)img->data[ic][py][px2] * kval;
              sum3 += (float)img->data[ic][py][px3] * kval;
            }
          }
        }
        
        if (sum0 < 0) sum0 = 0; else if (sum0 > img->max_val) sum0 = img->max_val;
        if (sum1 < 0) sum1 = 0; else if (sum1 > img->max_val) sum1 = img->max_val;
        if (sum2 < 0) sum2 = 0; else if (sum2 > img->max_val) sum2 = img->max_val;
        if (sum3 < 0) sum3 = 0; else if (sum3 > img->max_val) sum3 = img->max_val;
        
        result->data[oc][y][x]   = (unsigned char)sum0;
        result->data[oc][y][x+1] = (unsigned char)sum1;
        result->data[oc][y][x+2] = (unsigned char)sum2;
        result->data[oc][y][x+3] = (unsigned char)sum3;
      }
      
      // Handle remaining pixels
      for (; x < img->width; x++) {
        float sum = 0.0;
        
        for (int ic = 0; ic < kernel->input_channels; ic++) {
          for (int ky = 0; ky < kernel->height; ky++) {
            int py = y + ky - offset_y;
            if (py < 0) py = 0;
            else if (py >= img->height) py = img->height - 1;
            
            for (int kx = 0; kx < kernel->width; kx++) {
              int px = x + kx - offset_x;
              if (px < 0) px = 0;
              else if (px >= img->width) px = img->width - 1;
              
              sum += (float)img->data[ic][py][px] * kernel->data[oc][ic][ky][kx];
            }
          }
        }
        
        if (sum < 0) sum = 0;
        else if (sum > img->max_val) sum = img->max_val;
        result->data[oc][y][x] = (unsigned char)sum;
      }
    }
  }

  double end = omp_get_wtime();
  *time_taken = end - start;

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
    // Directory mode: parallel batch processing
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
    printf("Using %d OpenMP threads\n", omp_get_max_threads());

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

      int batch_size = batch_end - batch_start;
      printf("\nBatch %d-%d:\n", batch_start + 1, batch_end);

      Image **loaded_images = (Image **)calloc(batch_size, sizeof(Image *));
      char **filenames = (char **)calloc(batch_size, sizeof(char *));
      
      // Load all images first (not timed)
      for (int i = 0; i < batch_size; i++) {
        int idx = batch_start + i;
        filenames[i] = get_filename(image_files[idx]);
        printf("  [%d/%d] %s... ", idx + 1, num_images, filenames[i]);
        fflush(stdout);
        
        loaded_images[i] = read_image(image_files[idx]);
        if (!loaded_images[i]) {
          printf("FAILED (read)\n");
        } else {
          printf("loaded\n");
        }
      }

      double batch_conv_start = omp_get_wtime();
      
      #pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < batch_size; i++) {
        if (!loaded_images[i]) continue;
        
        Image *current = loaded_images[i];
        int success = 1;

        for (int k = 0; k < kset->num_kernels; k++) {
          double kernel_time = 0.0;
          Image *result =
              apply_convolution(current, kset->kernels[k], &kernel_time);

          if (!result) {
            if (current != loaded_images[i])
              free_image(current);
            success = 0;
            break;
          }

          if (current != loaded_images[i]) {
            free_image(current);
          }
          current = result;
        }

        if (success && current != loaded_images[i]) {
          loaded_images[i] = current;
        } else if (!success) {
          loaded_images[i] = NULL;
        }
      }
      
      double batch_conv_end = omp_get_wtime();
      double batch_conv_time = batch_conv_end - batch_conv_start;
      total_time += batch_conv_time;

      // Write all results (not timed)
      for (int i = 0; i < batch_size; i++) {
        int idx = batch_start + i;
        if (loaded_images[i]) {
          char *output_file = join_path(output_path, filenames[i]);
          if (write_image(output_file, loaded_images[i])) {
            processed++;
          } else {
            printf("  [%d/%d] %s... FAILED (write)\n", idx + 1, num_images, filenames[i]);
          }
          free(output_file);
          free_image(loaded_images[i]);
        }
        free(filenames[i]);
      }
      
      free(loaded_images);
      free(filenames);

      printf("Batch convolution time: %.6f seconds (%.2f images/sec)\n", 
             batch_conv_time, batch_size / batch_conv_time);
    }

    printf("\n=== Summary ===\n");
    printf("Processed: %d/%d images\n", processed, num_images);
    printf("Total computation time: %.6f seconds\n", total_time);
    printf("Average throughput: %.2f images/sec\n", processed / total_time);

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
    printf("Using %d OpenMP threads\n", omp_get_max_threads());

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