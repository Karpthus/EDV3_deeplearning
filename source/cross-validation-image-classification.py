import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from segment import maskPurpleBG  # Custom function to mask the purple background
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import itertools

# Path to dataset
data_path = r"C:\Users\stijn\Pictures\deeplearning"  # Replace with your dataset directory
batch_size = 72
image_shape = (224, 224, 3)
n_splits = 5  # Number of cross-validation folds

# Set-up data
file_list = [f for f in glob.iglob(os.path.sep.join([data_path, '**', '*.png']), recursive=True)
             if (os.path.isfile(f) and "annotated" not in f)]
print("[INFO] {} images found".format(len(file_list)))

# Label names based on directories in the dataset path
label_names = [os.path.split(f)[-1] for f in glob.iglob(os.path.sep.join([data_path, '*'])) if os.path.isdir(f)]

# Function to load and preprocess images
def load_and_preprocess_image(file_path, label):
    # Read image
    image = cv2.imread(file_path.decode('utf-8'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mask purple background
    mask = maskPurpleBG(image)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Resize and normalize
    masked_img = cv2.resize(masked_img, (224, 224))
    masked_img = masked_img.astype(np.float32) / 255.0
    
    return masked_img, label

# Prepare full dataset with labels
images = []
labels = []

# Collect images and labels
for label_idx, label_name in enumerate(label_names):
    label_path = os.path.join(data_path, label_name)
    label_files = glob.glob(os.path.join(label_path, '*.png'))
    
    for file in label_files:
        images.append(file)
        labels.append(label_idx)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation results storage
cv_scores = []

# Define the Hypermodel (same as before)
def build_model(hp):
    model = keras.Sequential()
    
    # First Conv Layer
    model.add(layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(224, 224, 3)
    ))
    model.add(layers.MaxPooling2D(2, 2))
    
    # Optional Dropout after the first Conv layer
    model.add(layers.Dropout(rate=hp.Float('dropout_conv1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Additional Conv Layers with Dropout
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i+2}', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(rate=hp.Float(f'dropout_conv_{i+2}', min_value=0.1, max_value=0.5, step=0.1)))

    # Flatten and Dense Layers with Dropout
    model.add(layers.Flatten())
    
    # Dense Layer
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    
    # Dropout after Dense Layer
    model.add(layers.Dropout(rate=hp.Float('dropout_dense', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output Layer
    model.add(layers.Dense(len(label_names), activation='softmax'))
    
    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(skf.split(images, labels), 1):
    print(f"\n[INFO] Fold {fold}")
    
    # Split data
    X_train, X_val = images[train_index], images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    # Convert to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(label_names))
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(label_names))
    
    # Create TensorFlow datasets
    def create_dataset(x, y, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        def py_func_preprocessing(file_path, label):
            img, _ = tf.py_function(
                load_and_preprocess_image, 
                [file_path, label], 
                [tf.float32, tf.int64]
            )
            img.set_shape((224, 224, 3))
            label_onehot = tf.one_hot(label, depth=len(label_names))
            return img, label_onehot
        
        dataset = dataset.map(py_func_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(x))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_dataset = create_dataset(X_train, y_train, is_training=True)
    val_dataset = create_dataset(X_val, y_val, is_training=False)
    
    # Instantiate the Hyperband tuner for this fold
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_hyperband',
        project_name=f'image_classification_fold_{fold}'
    )
    
    # Early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    
    # Hyperparameter search
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=[stop_early]
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build and train the best model for this fold
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )
    
    # Evaluate the model
    val_loss, val_accuracy = best_model.evaluate(val_dataset)
    cv_scores.append(val_accuracy)
    
    print(f"Fold {fold} - Validation Accuracy: {val_accuracy}")

# Print cross-validation results
print("\nCross-Validation Results:")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print("Individual Fold Accuracies:", cv_scores)

# Optional: Save the best model from the final fold
best_model.save('cross_validated_model.keras')

# Visualization of cross-validation results
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_splits+1), cv_scores)
plt.title('Cross-Validation Accuracies')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
