Cats vs Dogs Image Classification using SVM (Python)

This project implements a basic image classification system that distinguishes between cats and dogs using a Support Vector Machine (SVM). The model is trained on image data after converting each image into a numerical format.

The goal is to demonstrate how traditional machine learning techniques like SVM can be applied to image classification tasks.

+ Repository Structure
svm-cats-dogs-classifier/
│
├── svm_cats_dogs.py        → Main Python script
├── dataset/               → Dataset folder
│   ├── cats/              → Cat images
│   └── dogs/              → Dog images
└── README.md              → Project documentation


Note:
The full dataset is not uploaded due to size limits.
You should place your own images inside the dataset/cats and dataset/dogs folders.

+ Project Overview

Loads cat and dog images from folders

Resizes each image to 64 × 64

Converts images to RGB format

Flattens image data into 1D arrays

Splits data into training and testing sets

Trains a linear SVM classifier

Evaluates model accuracy

+ Technologies Used

Python

NumPy

Pillow (PIL)

Scikit-learn

+ How to Run the Project

Install dependencies:

pip install numpy pillow scikit-learn


Make sure your folder structure looks like this:

dataset/
  cats/
  dogs/


Run the script:

python svm_cats_dogs.py

+ Output

Console output showing:

Total images loaded

Model training status

Final accuracy score

Example:

Model training completed
Model Accuracy: 0.87

+ Use Case

This project is useful for:

Understanding image preprocessing

Learning how SVM works for classification

Practicing machine learning with real image data

+ Future Improvements

Add feature scaling

Use HOG or CNN features

Try different SVM kernels

Add model saving and loading



Note:
The dataset is not included due to size limits.
Download it from: <https://github.com/tanayshinde88-alt/CatsDogsProject.git>
Place it in the folder:
dataset/cats
dataset/dogs

