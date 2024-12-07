import os
import shutil
import cv2
import Augmentor

def augment_image(image_path, output_size=(244, 244), debug=False):
    """
    Augments a single image, saves it to the disk, and returns the augmented image as an OpenCV format (NumPy array).
    
    Args:
    - image_path (str): Path to the input image.
    - output_size (tuple): Desired output image size (width, height).
    - debug (bool): Whether to preview the augmentations (default is False).
    
    Returns:
    - augmented_image (np.array): The augmented image in OpenCV format (NumPy array).
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' does not exist. Please ensure the correct path.")

    # Create a temporary directory to hold the single image
    script_directory = os.path.dirname(os.path.abspath(__file__))
    temp_image_dir = os.path.join(script_directory, "temp_image_dir")
    os.makedirs(temp_image_dir, exist_ok=True)

    # Create an output directory for augmented images
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
        p.sample(5, multi_threaded=False)  # Preview 5 images for debugging
    else:
        # Generate a single augmented image (the pipeline will output at least one augmented image)
        p.sample(1, multi_threaded=False)

    # After augmentation, load the first augmented image from the output directory
    augmented_image_path = os.path.join(output_directory, os.listdir(output_directory)[0])

    # Load the augmented image using OpenCV (returns an array in BGR format)
    augmented_image = cv2.imread(augmented_image_path)

    # Delete the augmented image file after reading it into OpenCV
    if os.path.exists(augmented_image_path):
        os.remove(augmented_image_path)

    # Clean up temporary directories
    shutil.rmtree(temp_image_dir)

    # Return the augmented image in OpenCV format (NumPy array)
    return augmented_image

if __name__ == "__main__":
    # Define the folder containing the images
    image_folder = r'images\FiveCent'  # Update this path to the folder containing your images

    # Loop through each file in the directory
    for filename in os.listdir(image_folder):
        # Construct the full path of the image
        image_path = os.path.join(image_folder, filename)

        # Check if the file is an image (check extensions)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")

            # Augment the image
            augmented_image = augment_image(image_path, debug=False)

            # To visualize the augmented image (OpenCV)
            cv2.imshow("Augmented Image", augmented_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
