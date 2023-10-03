import numpy as np
import tensorflow as tf
import os
import cv2
from keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    MaxPool2D,
    Dropout,
    ReLU,
    AveragePooling2D,
)
from keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from edensenet_cls import EDenseNet as EDenseNet

print(tf.__version__)
seed_value = 1
tf.random.set_seed(seed_value)

# Model / data parameters
num_classes = 5
input_shape = (28, 28, 1)
batch_size = 16
epochs = 10
DATADIR = "gesture_dataset"
CATEGORIES = ["draw", "next", "point", "prev", "undo"]


# Load the data and split it between train and test sets
def load_data():
    x_data = []
    y_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(
                img_path, cv2.IMREAD_GRAYSCALE
            )  # Load images in grayscale
            img_array = cv2.resize(img_array, (28, 28))
            # plt.imshow(img_array, cmap="gray")
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


class Model_Parameters:
    num_layers = 3
    growth_rate = 24
    filter_size = 3
    fm_1st_layer = 32
    bottleneck = 4
    dropout_prob = 0.2  # not 0.8, 0.8 drops 80% of the elements
    learn_rate = 0.001
    num_classes = 6


param = Model_Parameters()
model = EDenseNet(param, name="EDenseNet")
model = model.model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=param.learn_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)
score = model.evaluate(x_test, y_test, verbose=0)
model.save("experiment_0911.h5")
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

# Show the plot
plt.show()
