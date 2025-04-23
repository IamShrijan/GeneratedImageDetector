import pandas as pd

from PIL import Image
import os

def preprocess_images(input_dir: str, output_dir: str, target_size=(768, 512)):
    """
    Preprocess images to a specific size by resizing, padding, or cropping them.

    Parameters:
    input_dir (str): Directory containing the input images.
    output_dir (str): Directory where processed images will be saved.
    target_size (tuple): Desired size of the output images (width, height).

    Returns:
    None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Ensure image is in RGB format

                # Resize image while maintaining aspect ratio
                img.thumbnail(target_size, Image.ANTIALIAS)

                # Create a new image with the target size and a white background
                new_img = Image.new('RGB', target_size, (255, 255, 255))
                new_img.paste(
                    img, 
                    ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
                )

                # Save the processed image
                new_img.save(os.path.join(output_dir, filename))

