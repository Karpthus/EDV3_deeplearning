import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import cv2
import numpy as np
import os
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from segment import maskPurpleBG  # Import your custom masking function

# Settings
data_path = r"C:\Users\stijn\Pictures\deeplearning"  # Replace with your dataset path
image_shape = (224, 224, 3)
batch_size = 72
model_path = 'best_model.keras'  # Path to the saved model
num_images_per_class = 50  # Limit per label

# Load the label names based on directories
label_names = [os.path.split(f)[-1] for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
print(f"[INFO] Labels: {label_names}")

# Image Preprocessing Function
def preprocess_image_opencv(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = maskPurpleBG(image_bgr)  # Apply purple background masking
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    masked_img_rgb = masked_img_rgb.astype(np.float32) / 255.0  # Normalize
    return masked_img_rgb

# TensorFlow preprocessing wrapper
def preprocess_image(image, label):
    image = tf.numpy_function(preprocess_image_opencv, [image], tf.float32)
    image.set_shape((224, 224, 3))
    return image, label

# Function to get limited images per class
def get_limited_data(generator, num_per_class):
    """
    Extracts a limited number of images per class.
    """
    class_indices = generator.class_indices
    files_by_class = {class_name: [] for class_name in class_indices}

    # Collect file paths grouped by class
    for file, label in zip(generator.filepaths, generator.labels):
        class_name = list(class_indices.keys())[label]  # Convert dict_keys to list and access class name
        files_by_class[class_name].append(file)

    # Sample exactly 'num_per_class' images per class
    sampled_files = []
    sampled_labels = []
    for class_name, file_list in files_by_class.items():
        sampled_files.extend(file_list[:num_per_class])
        sampled_labels.extend([class_indices[class_name]] * num_per_class)

    return sampled_files, sampled_labels

# Load Validation Dataset
print("[INFO] Loading data...")

data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Use flow_from_directory to get all validation data
full_val_generator = data_gen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=1,  # Load one image at a time
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Keep order for sampling
)

# Sample 50 images per class
sampled_files, sampled_labels = get_limited_data(full_val_generator, num_images_per_class)

# Create a limited dataset
def limited_data_generator(files, labels):
    for file, label in zip(files, labels):
        image = keras.utils.load_img(file, target_size=(224, 224))
        image = keras.utils.img_to_array(image) / 255.0  # Rescale
        yield image, tf.keras.utils.to_categorical(label, num_classes=len(label_names))

# Create the dataset
limited_dataset = tf.data.Dataset.from_generator(
    lambda: limited_data_generator(sampled_files, sampled_labels),
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(label_names),), dtype=tf.float32)
    )
).map(preprocess_image).batch(batch_size, drop_remainder=False)

# Load the Model
print("[INFO] Loading the model...")
model = load_model(model_path)

# Collect Predictions and True Labels
val_labels = []  # True labels
val_preds = []   # Predicted labels

print("[INFO] Generating predictions...")
for images, labels in limited_dataset:
    predictions = model.predict(images, verbose=0)  # Make predictions
    val_preds.extend(np.argmax(predictions, axis=1))  # Predicted class indices
    val_labels.extend(np.argmax(labels.numpy(), axis=1))  # True class indices

# Confusion Matrix
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(val_labels, val_preds)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Plot and Save Confusion Matrix
plot_confusion_matrix(cm, classes=label_names, title="Confusion Matrix")

# Print Confusion Matrix
print("Confusion Matrix:")
print(cm)
