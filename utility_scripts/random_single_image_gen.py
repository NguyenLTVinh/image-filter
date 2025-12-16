import numpy as np
import random
import os

def generate_ppm_image(filename, width=4096, height=4096, max_val=255):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Generate the whole image at once: shape (height, width, 3)
    img = np.random.randint(0, max_val + 1, size=(height, width, 3), dtype=int)
    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n{max_val}\n")
        # Write each row as a single line
        for y in range(height):
            row = img[y].reshape(-1)
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')
    print(f"PPM image '{filename}' generated successfully.")

w = random.randint(256, 4096)
h = random.randint(256, 4096)

for i in range(24, 25):
    generate_ppm_image(f"tests/test_{i}/in.ppm", width=w, height=h, max_val=255)
