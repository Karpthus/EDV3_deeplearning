from augmentation import augment_image
import numpy as np
# import tensorflow as tf
import cv2
import os

def preprocess_image(image, kernel_type="sharpen", resize_dim=(224, 224), apply_hist_eq=False, apply_gaussian_blur=True):
    kernels = {
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "blur": np.ones((3, 3), np.float32) / 9.0,
    }
    kernel = kernels.get(kernel_type, kernels["sharpen"])

    try:
        # Step 1: Resize the image (default to 224x224)
        image = cv2.resize(image, resize_dim)

        # Step 2: Apply filter (sharpening or blurring)
        image = cv2.filter2D(image, -1, kernel)

        # Step 3: Histogram Equalization (for contrast enhancement, applicable only on grayscale images)
        if apply_hist_eq and len(image.shape) == 2:  # Check if image is grayscale
            image = cv2.equalizeHist(image)

        # Step 4: Apply Gaussian blur (optional)
        if apply_gaussian_blur:
            image = cv2.GaussianBlur(image, (5, 5), 0)

        # Step 5: Normalize the image (convert pixel values to the range [0, 1])
        image = image.astype('float32') / 255.0

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    return image


if __name__ == "__main__":
    image_folder = r'images\FiveCent'  # Update this path to the folder containing your images

    # Loop through each file in the directory
    for filename in os.listdir(image_folder):
        # Construct the full path of the image
        image_path = os.path.join(image_folder, filename)

        # Check if the file is an image (check extensions)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            image = augment_image(image_path)

            # Step 2: Preprocess the augmented image (e.g., apply sharpening)
            processed_image = preprocess_image(image)

            if processed_image is not None:
                # Step 3: Display the preprocessed image (scaled to [0, 255] for OpenCV)
                processed_image = (processed_image * 255).astype(np.uint8)  # Scale back to 0-255
                cv2.imshow('original', image)
                cv2.imshow('Processed Image', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Image preprocessing failed.")