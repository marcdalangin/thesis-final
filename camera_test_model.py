import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model("model/trained_model_0919.h5")

# Define the class labels for hand gestures
CATEGORIES = ["1", "2", "3", "4", "5"]


# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return image


# Function to predict the hand gesture
def predict_hand_gesture(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.array([processed_image]))
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CATEGORIES[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class_label, confidence


# Capture video from the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for background subtraction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Define a region of interest (ROI) for the hand
    roi = blurred_frame[50:300, 50:300]

    # Get the predicted label and confidence for the hand gesture
    predicted_label, confidence = predict_hand_gesture(roi)

    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (hand)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Adjust the coordinates based on the ROI's position
        x += 50
        y += 50

        # Draw a bounding box around the hand region in green
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted label and confidence on the frame
        text = f"Predicted Gesture: {predicted_label} (Confidence: {confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
