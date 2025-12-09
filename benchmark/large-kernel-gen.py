import numpy as np

def generate_gaussian_kernel(size, sigma):
    """Generate a normalized 2D Gaussian kernel."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel

def save_3d_kernel(filename, size, sigma, input_channels=3, output_channels=3):
    """
    Save a 3D convolution kernel for image filtering.
    For standard blur: diagonal structure (R→R, G→G, B→B)
    """
    kernel_2d = generate_gaussian_kernel(size, sigma)

    with open(filename, 'w') as f:
        f.write("1\n")
        f.write(f"{size} {size} {input_channels} {output_channels}\n")

        for oc in range(output_channels):
            for ic in range(input_channels):
                if oc == ic:
                    for row in kernel_2d:
                        f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
                else:
                    for _ in range(size):
                        f.write(" ".join("0.000000" for _ in range(size)) + "\n")

def print_kernel_info(size, sigma):
    """Print information about the generated kernel."""
    kernel = generate_gaussian_kernel(size, sigma)
    print(f"Generated {size}x{size} Gaussian kernel with σ={sigma}")
    print(f"Center value: {kernel[size//2, size//2]:.6f}")
    print(f"Sum of all values: {np.sum(kernel):.10f}")
    print(f"Min value: {np.min(kernel):.6f}")
    print(f"Max value: {np.max(kernel):.6f}")

kernel_size = 63
sigma = 2.0
input_channels = 3
output_channels = 3
output_filename = "large-kernel.txt"

if __name__ == "__main__":
    print_kernel_info(kernel_size, sigma)
    save_3d_kernel(output_filename, kernel_size, sigma, input_channels, output_channels)
    print(f"Kernel saved to: {output_filename}")
