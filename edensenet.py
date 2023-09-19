import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.utils import plot_model
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Model / data parameters
num_classes = 5  # Number of classes in your dataset
input_shape = (28, 28, 3)
batch_size = 16
epochs = 13
DATADIR = "train_dataset"
CATEGORIES = ["1", "2", "3", "4", "5"]
NAME = "edensenet{}".format(epochs)


# Load the data and split it between train and test sets
def load_data():
    x_data = []
    y_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (28, 28))
            plt.imshow(img_array, cmap="gray")
            # print(img_array)
            x_data.append(img_array)
            y_data.append(class_label)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


x_train, y_train = load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print(x_test.shape[0], "test samples")

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Train the model
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)
score = model.evaluate(x_test, y_test, verbose=0)
model.save(
    "model/trained_model_0919.h5"
)  # Rename model with this format "trained_model_YYYYMMDDvX.h5"

plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    show_trainable=True,
)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Plot the training history
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.title("Training History")

# Save the figure as a .jpg file
plt.savefig("training_history.jpg")
