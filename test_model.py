import tensorflow as tf
import cv2
import numpy as np
import keras

# Load your trained model (Latest trained model should be used here)
model = keras.models.load_model("model/trained_model_1003.h5")

# Define the class labels for hand gestures
CATEGORIES = ["1", "2", "3", "4", "5"]


# # Function to preprocess a single input image
# def preprocess_image(image):
#     # Resize the image to the input shape of the model
#     image = cv2.resize(image, (28, 28))
#     # Convert the image to grayscale
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Normalize the image
#     image = image.astype("float32") / 255.0
#     # Expand dimensions to make it (28, 28, 1)
#     image = np.expand_dims(image, axis=-1)
#     return image


# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return image


# Function to predict hand gestures for multiple images
def predict_hand_gestures(images):
    predictions = []

    for image in images:
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make the prediction
        prediction = model.predict(np.array([processed_image]))
        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        # Get the predicted class label
        predicted_class_label = CATEGORIES[predicted_class_index]
        # Get the confidence score of the prediction
        confidence = prediction[0][predicted_class_index]
        predictions.append((predicted_class_label, confidence))

    return predictions


# Load and process multiple user input images
image_paths = [
    "validation_dataset/1/Sign_1_46.png",
    "validation_dataset/2/IMG_4101.JPG",
    "validation_dataset/3/Sign_3_40.png",
    "validation_dataset/4/IMG_4840.JPG",
    "validation_dataset/5/Sign_5_24.png",
    # Add more image paths as needed
]

# Load the correct labels for validation images
correct_labels = [image_path.split("/")[1] for image_path in image_paths]

user_input_images = [cv2.imread(image_path) for image_path in image_paths]
user_input_images = [cv2.resize(image, (28, 28)) for image in user_input_images]
predicted_labels_and_confidences = predict_hand_gestures(user_input_images)

# Display the predicted labels, correct labels, and confidences for each image
for i, (predicted_label, confidence) in enumerate(predicted_labels_and_confidences):
    correct_label = correct_labels[i]
    confidence_percentage = f"{confidence * 100:.2f}%"
    print(
        f"Image {i + 1}: Predicted Gesture: {predicted_label}, Correct Gesture: {correct_label}, Confidence: {confidence_percentage}"
    )

# Calculate the average confidence rate
confidence_rates = [confidence for _, confidence in predicted_labels_and_confidences]
average_confidence = np.mean(confidence_rates)

# Print the average confidence rate
average_confidence_percentage = f"{average_confidence * 100:.2f}%"
print(f"Average Confidence Rate: {average_confidence_percentage}")
