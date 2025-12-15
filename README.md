This program applies convolution kernels to PPM (color) or PGM (grayscale) images.

```
./filter-serial <input.ppm|input.pgm> <kernel.txt> <output.ppm|output.pgm>
```

## Inputs

### Input Image Format
- **PPM (P3)**: Color image (RGB, 3 channels).
- **PGM (P2)**: Grayscale image (1 channel).
- **Header**: Format identifier (`P3` or `P2`), followed by width, height, and max pixel value.

---

### Kernel File Format
- **Line 1**: Number of kernels.
- **For each kernel**:
  - **Line 1**: `<height> <width> <input_channels> <output_channels>`
  - **Following lines**: Kernel values (for each output channel, for each input channel, row-major order).

**Note**: Kernel dimensions must be odd.

## Tensor Representation

### Image as a 3D Tensor
- An image is represented as a **3D tensor** with dimensions:
  - **Height (H)**: Number of rows (vertical pixels).
  - **Width (W)**: Number of columns (horizontal pixels).
  - **Channels (C)**: Number of color channels (e.g., 3 for RGB, 1 for grayscale).

- For example, an RGB image of size 640x480 is a tensor of shape `(480, 640, 3)`:
  - `480` rows (height),
  - `640` columns (width),
  - `3` channels (Red, Green, Blue).

- In the code, the image tensor is stored as `unsigned char ***data`, where:
  - `data[channel][height][width]` accesses the pixel value at `(height, width)` for a specific channel.

### Kernel as a 4D Tensor
- A convolution kernel is represented as a **4D tensor** with dimensions:
  - **Output Channels (K)**: Number of output feature maps.
  - **Input Channels (C)**: Number of input channels (must match the image's channels).
  - **Kernel Height (kh)**: Vertical size of the kernel.
  - **Kernel Width (kw)**: Horizontal size of the kernel.

- For example, a kernel that converts a 3-channel RGB image to a 1-channel grayscale image (e.g., edge detection) has shape `(1, 3, kh, kw)`.

- In the code, the kernel tensor is stored as `float ****data`, where:
  - `data[out_ch][in_ch][ky][kx]` accesses the kernel weight for:
    - `out_ch`: Output channel,
    - `in_ch`: Input channel,
    - `(ky, kx)`: Position within the kernel.

## Testing

The program can be tested for correctness and speedups by running:

```
./test.sh <program.cu>
```

The script runs program.cu and compares its output with the expected output, passing the test if the two match. The speedup tests pass as long as the program is at least 5% faster than the baseline serial program. Each test folder contains `in.ppm` (image), `in.txt` (kernel), and `out.ppm` (expected output).

Tests:
- 0: Blur
- 1: Edge Detect Per Channel
- 2: Gaussian Blur
- 3: Grayscale
- 4: Sepia
- 5: YUV
- 6–14: Generated correctness tests
- 15–24: Generated speedup tests
