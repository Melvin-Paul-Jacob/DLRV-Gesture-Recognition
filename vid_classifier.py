import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Specify the path to your dataset directory
dataset_path = "/path/to/your/dataset"

# Define the categories (classes) for your video classifier
categories = ['category1', 'category2', 'category3']

# Load and preprocess the video frames
def load_frames(video_path):
    frames = []
    for frame_file in sorted(os.listdir(video_path)):
        img = image.load_img(os.path.join(video_path, frame_file), target_size=(64, 64))
        img = image.img_to_array(img)
        frames.append(img)
    return np.array(frames)

# Load the dataset and preprocess the data
def load_dataset():
    X = []
    y = []
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for video in os.listdir(category_path):
            video_path = os.path.join(category_path, video)
            frames = load_frames(video_path)
            X.append(frames)
            y.append(categories.index(category))
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y)
    return X, y

# Split the dataset into training and testing sets
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Build the video classifier model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(categories), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the dataset
X, y = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Build the video classifier model
model = build_model()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
