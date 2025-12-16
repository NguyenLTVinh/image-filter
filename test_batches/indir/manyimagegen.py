import numpy as np

def generate_ppm_image(filename, width=4096, height=4096, max_val=255):
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

def generate_multiple_ppm_images(num_images, base_filename="large-image",
                                  width=1024, height=1024, max_val=255):
    for i in range(num_images):
        filename = f"{base_filename}_{i+1}.ppm"
        generate_ppm_image(filename, width, height, max_val)
    print(f"\nAll {num_images} images generated successfully.")

generate_multiple_ppm_images(num_images=1024, width=200, height=200, max_val=255)
