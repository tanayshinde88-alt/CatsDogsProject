print("Program started...")

import os
import numpy as np
from PIL import Image

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# Basic settings
# -----------------------------
data_path = "dataset"
image_size = 64
max_images = 500

image_data = []
image_labels = []


# -----------------------------
# Load cat and dog images
# -----------------------------
folders = ["cats", "dogs"]

for index in range(len(folders)):
    folder_name = folders[index]
    folder_location = os.path.join(data_path, folder_name)

    file_list = os.listdir(folder_location)[:max_images]
    print(folder_name, "images found:", len(file_list))

    for file in file_list:
        file_path = os.path.join(folder_location, file)

        try:
            image = Image.open(file_path)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))

            image_array = np.array(image).flatten()

            image_data.append(image_array)
            image_labels.append(index)

        except:
            # Ignore corrupted images
            continue


print("Total images loaded:", len(image_data))


# -----------------------------
# Convert lists to arrays
# -----------------------------
X = np.array(image_data)
y = np.array(image_labels)

# Safety check
if len(set(y)) < 2:
    print("Error: Both classes not found in dataset")
    input("Press Enter to exit...")
    exit()


# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Train SVM model
# -----------------------------
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

print("Model training completed")


# -----------------------------
# Test model
# -----------------------------
predicted_values = svm_model.predict(X_test)
model_accuracy = accuracy_score(y_test, predicted_values)

print("Model Accuracy:", model_accuracy)


input("Press Enter to exit...")
