import os
import shutil
import cv2
import Augmentor

def augment_image(image_path, batch_size=5, output_size=(244, 244), debug=False):
    """
    Augments a single image and saves a specified batch size of augmented images to a common output folder.
    
    Args:
    - image_path (str): Path to the input image.
    - batch_size (int): Number of augmented images to generate.
    - output_size (tuple): Desired output image size (width, height).
    - debug (bool): Whether to preview the augmentations (default is False).
    
    Returns:
    - None: Saves the augmented images in the common augmented_images directory.
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' does not exist. Please ensure the correct path.")
    
    # Create a temporary directory to hold the single image
    script_directory = os.path.dirname(os.path.abspath(__file__))
    temp_image_dir = os.path.join(script_directory, "temp_image_dir")
    os.makedirs(temp_image_dir, exist_ok=True)

    # Create a common output directory for all augmented images
    output_directory = os.path.join(script_directory, "augmented_images")
    os.makedirs(output_directory, exist_ok=True)

    # Copy the image into the temporary directory
    shutil.copy(image_path, temp_image_dir)

    # Set up the Augmentor pipeline for the temporary directory (which contains the single image)
    p = Augmentor.Pipeline(source_directory=temp_image_dir, output_directory=output_directory)

    # Define augmentations
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.4)
    p.flip_top_bottom(probability=0.4)
    p.rotate_random_90(probability=0.75)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.scale(probability=0.5, scale_factor=1.2)
    p.skew(probability=0.1, magnitude=0.5)
    p.random_color(probability=0.6, min_factor=0.6, max_factor=1.5)
    p.random_brightness(probability=0.6, min_factor=0.6, max_factor=1.5)
    p.random_contrast(probability=0.6, min_factor=0.6, max_factor=1.5)
    p.resize(probability=1, width=output_size[0], height=output_size[1], resample_filter="BICUBIC")

    # In debug mode, preview the augmentations
    if debug:
        print("Debug mode enabled. Previewing augmentations...")
        p.sample(batch_size, multi_threaded=False)  # Preview batch_size images for debugging
    else:
        # Generate the specified number of augmented images (the pipeline will output the same number)
        p.sample(batch_size, multi_threaded=False)

    # Clean up temporary directories
    shutil.rmtree(temp_image_dir)

    # Rename the generated images with unique names based on the original image
    filename = os.path.basename(image_path)
    file_name_without_extension = os.path.splitext(filename)[0]

    # Iterate through the augmented images and rename them
    augmented_images = os.listdir(output_directory)
    index = 1
    for img in augmented_images:
        if img.endswith(('.png', '.jpg', '.jpeg')):
            # Rename and save augmented images with an index appended
            new_name = f"{file_name_without_extension}_{index}.jpg"
            old_path = os.path.join(output_directory, img)
            new_path = os.path.join(output_directory, new_name)
            os.rename(old_path, new_path)
            index += 1

    print(f"Saved {batch_size} augmented images to {output_directory}")

if __name__ == "__main__":
    # Define the folder containing the images
    image_folder = r'images\TwoEuro'  # Update this path to the folder containing your images
    batch_size = 20  # Specify the number of augmented images to generate for each image

    # Loop through each file in the directory
    for filename in os.listdir(image_folder):
        # Construct the full path of the image
        image_path = os.path.join(image_folder, filename)

        # Check if the file is an image (check extensions)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")

            # Augment the image and save a batch of augmented images in the common folder
            augment_image(image_path, batch_size=batch_size, debug=False)

    print("Image augmentation completed!")
