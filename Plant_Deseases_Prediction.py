import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import label_binarize, LabelBinarizer
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.utils import to_categorical 
# Define the root directory
root_dir = Path("./Data")

# Define the list of labels
all_labels = ['Corn_(maize)___Common_rust_', 'Potato___Early_blight', 'Tomato___Bacterial_spot']


# Define a function to convert an image to an array
def convert_image_to_array(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            img_array = np.array(img, dtype=np.float16) / 225.0
            return img_array
    except Exception as e:
        print(f"Error : {e}")
        return None

# Loop through the directories and load the images and labels
image_list, label_list = [], []
for i, directory in enumerate(root_dir.glob("*")):
    for image_path in directory.glob("*"):
        image_list.append(convert_image_to_array(image_path))
        label_list.append(i)

# Convert the lists to arrays
image_array = np.array(image_list)
label_array = np.array(label_list)

# Define the list of binary labels
binary_labels = label_binarize(label_array, classes=[0, 1, 2])

# Visualize the number of classes count
label_counts = pd.DataFrame(label_array).value_counts()
print(label_counts.head())

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(image_array, label_array, test_size=0.2, random_state=10)

# Split the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Convert the labels to categorical
num_classes = len(all_labels)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)


# Define the model architecture
model = Sequential()

# Adding the first Convolutional layer
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Adding the second Convolutional layer
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Adding the third Convolutional layer
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output of the last Convolutional layer
model.add(Flatten())

# Adding the first dense layer
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))

# Adding the output layer
model.add(Dense(num_classes, activation="softmax"))

# Printing summary of the model
model.summary()

#Define the optimizer and compile the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

#Set up early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)

#Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stop])

#Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")

#Plot the training and validation accuracy and loss
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

model.save("./model/plant_disease.h5")
# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('./model/plant_model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('./model/plant_model_weights.h5')