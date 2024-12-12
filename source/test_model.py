import cv2
import tensorflow as tf
import numpy as np
from segment import maskPurpleBG  # Custom function for purple background masking

# Load the trained Keras model
model_path = 'best_model.keras'  # Path to the saved model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Parameters
image_shape = (224, 224)  # Input size for the model
confidence_threshold = 0.80  # Minimum confidence to display prediction
output_video_path = 'output.avi'  # Path for recorded video file

# Function to preprocess each frame
def preprocess_frame(image):
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

# OpenCV: Capture video from the webcam
cap = cv2.VideoCapture(1)  # 0 for default webcam, or use a video file path

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Initialize video writer for recording
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 fps if not detected
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Label names for predictions
label_names = ['FiveCent', 'OneEuro', 'TwoEuro']  # Replace with actual class names

# Process video feed
print("Starting live video feed. Press 'q' to quit.")
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(preprocessed_frame, verbose=0)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    # Check confidence threshold
    if confidence >= confidence_threshold:
        predicted_class = label_names[predicted_class_index]
    else:
        predicted_class = "Unknown"

    # Display the prediction on the frame
    text = f"Prediction: {predicted_class} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame with prediction
    cv2.imshow("Live Feed - Model Predictions", frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Close the video writer
cv2.destroyAllWindows()
print("Video feed stopped. Video saved to:", output_video_path)