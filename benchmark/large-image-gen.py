def generate_ppm_image(filename, width=4096, height=4096, max_val=255):
    """
    Generate a large PPM (P3) color image with random pixel values.

    Args:
        filename (str): Output filename.
        width (int): Image width.
        height (int): Image height.
        max_val (int): Maximum color value (default: 255).
    """
    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n{max_val}\n")

        for y in range(height):
            for x in range(width):
                r = str(int(np.random.randint(0, max_val + 1)))
                g = str(int(np.random.randint(0, max_val + 1)))
                b = str(int(np.random.randint(0, max_val + 1)))
                f.write(f"{r} {g} {b} ")
            f.write("\n")

    print(f"PPM image '{filename}' generated successfully.")

import numpy as np
generate_ppm_image("large-image.ppm", width=1024, height=1024, max_val=255)
