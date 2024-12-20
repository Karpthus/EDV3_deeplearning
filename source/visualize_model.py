import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from segment import maskPurpleBG  # Custom function to mask the purple background

# === Paths to the Model and Image === #
model_path = 'best_model.keras'  # Path to the trained model
test_image_path = r"C:\Users\stijn\Pictures\deeplearning\FiveCent\1726687703130446000.png"  # Path to a test image for visualization

# === Preprocessing Logic === #
def preprocess_image_opencv(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    mask = maskPurpleBG(image)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    masked_img_rgb = cv2.resize(masked_img_rgb, (224, 224))
    masked_img_rgb = masked_img_rgb.astype(np.float32) / 255.0
    return masked_img_rgb

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model input."""
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize to 224x224
    image = preprocess_image_opencv(image)  # Apply custom preprocessing
    image_tensor = np.expand_dims(image, axis=0)  # Add batch dimension
    return image_tensor

# === Visualize Feature Maps === #
def display_feature_maps(activations, layer_names, max_features=6):
    """
    Display feature maps for intermediate layers.
    - activations: List of activation outputs.
    - layer_names: Corresponding layer names.
    - max_features: Maximum number of feature maps to display per layer.
    """
    for layer_name, activation in zip(layer_names, activations):
        n_features = activation.shape[-1]  # Number of feature maps
        size = activation.shape[1]  # Width/height of feature maps

        print(f"\nVisualizing layer: {layer_name} ({n_features} features)")

        # Display up to 'max_features' maps per layer
        fig, axes = plt.subplots(1, min(n_features, max_features), figsize=(20, 5))
        fig.suptitle(f"Layer: {layer_name}", fontsize=16)

        for i in range(min(n_features, max_features)):
            ax = axes[i]
            feature_map = activation[0, :, :, i]  # Extract feature map for the first image
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')

        plt.show()

# === Main Script === #
if __name__ == "__main__":
    # 1. Load the trained model
    print("[INFO] Loading the trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = load_model(model_path)

    # 2. Display model summary
    print("\n[INFO] Model Summary:")
    model.summary()

    # 3. Select convolutional layers to visualize
    conv_layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    print("\n[INFO] Selected layers for visualization:")
    print(conv_layer_names)

    # 4. Create a model to output activations for selected layers
    layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

    # 5. Load and preprocess the test image
    print("\n[INFO] Preprocessing the test image...")
    image_tensor = load_and_preprocess_image(test_image_path)

    # 6. Generate activations
    print("\n[INFO] Generating activations...")
    activations = activation_model.predict(image_tensor)

    # 7. Display feature maps
    print("\n[INFO] Displaying feature maps...")
    display_feature_maps(activations, conv_layer_names)

    print("\n[INFO] Visualization complete.")
