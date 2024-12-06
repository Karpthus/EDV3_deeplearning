import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner as kt
import datetime
import matplotlib.pyplot as plt

# Set directory paths
data_dir = r"C:\Git\EDV3_deeplearning\images"  # Replace with your dataset directory

# Load dataset
batch_size = 8
img_height = 224  # Resize all images to 224x224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,  # Ensure reproducibility
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Class names
class_names = train_ds.class_names
print("Class Names:", class_names)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Clear any previous session to avoid memory overflow (useful when using GPUs)
tf.keras.backend.clear_session()

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),  # Define input shape explicitly
        layers.Rescaling(1./255),  # Normalize pixel values
        
        # First convolutional block
        layers.Conv2D(
            filters=hp.Int('filters_layer1', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(
            filters=hp.Int('filters_layer2', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(hp.Float('dropout_conv', min_value=0.1, max_value=0.5, step=0.1)),

        layers.Flatten(),
        
        # Dense layer
        layers.Dense(
            units=hp.Int('units_dense', min_value=128, max_value=512, step=64),
            activation='relu'
        ),
        layers.Dropout(hp.Float('dropout_dense', min_value=0.1, max_value=0.5, step=0.1)),
        
        # Output layer
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Instantiate the Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='my_dir',
    project_name='custom_image_training_hyperband'
)

# Display search space summary
tuner.search_space_summary()

# Run the hyperparameter search
try:
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[tensorboard_callback]
    )
except Exception as e:
    print(f"Error during hyperparameter search: {e}")

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Optimal filters for first conv layer: {best_hps.get('filters_layer1')}
Optimal filters for second conv layer: {best_hps.get('filters_layer2')}
Optimal units in dense layer: {best_hps.get('units_dense')}
Optimal dropout rate in conv layers: {best_hps.get('dropout_conv')}
Optimal dropout rate in dense layers: {best_hps.get('dropout_dense')}
Optimal learning rate: {best_hps.get('learning_rate')}
""")

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)

try:
    history = best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[tensorboard_callback]
    )
except Exception as e:
    print(f"Error during training: {e}")

# Evaluate the model
try:
    loss, accuracy = best_model.evaluate(val_ds)
    print(f"Validation Accuracy: {accuracy:.2f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Visualize training history
try:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()
except Exception as e:
    print(f"Error during plotting: {e}")
