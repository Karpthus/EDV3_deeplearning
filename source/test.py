from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set up ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
test_datagen = ImageDataGenerator(rescale=1./255)

# Directory where the data is located
data_dir = 'images'

# Flow training images from the directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Resize all images to 28x28 pixels
    batch_size=32,  # Number of images to yield per batch
    class_mode='sparse',  # Use sparse labels for integer encoding
    color_mode='grayscale',  # Since Fashion MNIST is grayscale
)

# Flow testing images from the directory
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Use sparse labels for integer encoding
    color_mode='grayscale',  # Since Fashion MNIST is grayscale
)

# Now you can use the train_generator and test_generator for model training and evaluation

from tensorflow import keras
from tensorflow.keras import layers

# Define a simple model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # sparse labels are used
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
