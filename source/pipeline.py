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
import itertools



# Value             |Best Value So Far |Hyperparameter
# 128               |64                |filters_1
# 1                 |2                 |num_conv_layers
# 64                |32                |filters_2
# 64                |256               |dense_units
# 0.01              |0.001             |learning_rate
# 32                |64                |filters_3
# 32                |96                |filters_4
# 0.2               |0.4               |dropout_conv1
# 0.2               |0.1               |dropout_conv_2
# 0.2               |0.4               |dropout_dense
# 0.1               |0.2               |dropout_conv_3
# 10                |10                |tuner/epochs
# 0                 |4                 |tuner/initial_epoch
# 0                 |1                 |tuner/bracket
# 0                 |1                 |tuner/round


# Path to dataset
data_path = r"C:\Users\stijn\Pictures\deeplearning"  # Replace with your dataset directory
batch_size = 72
image_shape = (224, 224, 3)

# Set-up data
file_list = [f for f in glob.iglob(os.path.sep.join([data_path, '**', '*.png']), recursive=True)
             if (os.path.isfile(f) and "annotated" not in f)]
print("[INFO] {} images found".format(len(file_list)))

# Label names based on directories in the dataset path
label_names = [os.path.split(f)[-1] for f in glob.iglob(os.path.sep.join([data_path, '*'])) if os.path.isdir(f)]

# ImageDataGenerator for data augmentation and normalization
data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% for validation
)

# Function to preprocess image using OpenCV
def preprocess_image_opencv(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = maskPurpleBG(image_bgr)
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    masked_img_rgb = masked_img_rgb.astype(np.float32) / 255.0
    return masked_img_rgb

# TensorFlow preprocessing
def preprocess_image(image, label):
    image = tf.numpy_function(preprocess_image_opencv, [image], tf.float32)
    image.set_shape((224, 224, 3))
    return image, label

# Create train and validation datasets
train_generator = data_gen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Convert to tf.data.Dataset and apply preprocessing
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(label_names)), dtype=tf.float32)
    )
).unbatch().map(preprocess_image).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(label_names)), dtype=tf.float32)
    )
).unbatch().map(preprocess_image).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

train_steps_per_epoch = len(train_generator)  # Number of batches per epoch for training
val_steps_per_epoch = len(val_generator)      # Number of batches per epoch for validation

# TensorBoard Callback Directory
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define the Hypermodel
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


# Instantiate the Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_hyperband',
    project_name='image_classification'
)

# Define early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# Run the hyperparameter search
print("[INFO] Starting Hyperparameter Tuning...")
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    callbacks=[stop_early, tensorboard_callback]
)

# Retrieve the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal hyperparameters:
- Filters for first layer: {best_hps.get('filters_1')}
- Number of Conv Layers: {best_hps.get('num_conv_layers')}
- Dense Units: {best_hps.get('dense_units')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# # Evaluate the best model on the validation dataset
# val_labels = []  # True labels
# val_preds = []   # Predicted labels

# # Loop through the validation dataset to collect predictions and true labels
# for images, labels in val_dataset:
#     predictions = best_model.predict(images)  # Make predictions
#     val_preds.extend(np.argmax(predictions, axis=1))  # Get predicted class indices
#     val_labels.extend(np.argmax(labels.numpy(), axis=1))  # Get true class indices

# # Generate confusion matrix
# cm = confusion_matrix(val_labels, val_preds)
# print("Confusion Matrix:")
# print(cm)

# # Plot confusion matrix as a heatmap
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
#     plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.show()

# # Call the function to plot the confusion matrix
# plot_confusion_matrix(cm, classes=label_names, title="Confusion Matrix")


# Save the best model
best_model.save('best_model1313.keras')

# Plot the results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Instructions to run TensorBoard
print(f"To view TensorBoard logs, run the following command in your terminal:")
print(f"tensorboard --logdir={os.path.abspath(log_dir)}")
