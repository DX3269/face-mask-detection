import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# Paths to the folders containing images

with_mask_path = r'C:\Users\Ayush Thakur\Desktop\projects\New folder\data\with_mask'
without_mask_path = r'C:\Users\Ayush Thakur\Desktop\projects\New folder\data\without_mask'

# List to store the annotations
annotations = []

# Processing images with masks
for filename in os.listdir(with_mask_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        annotations.append([filename, with_mask_path, 'mask'])

# Processing images without masks
for filename in os.listdir(without_mask_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        annotations.append([filename, without_mask_path, 'no_mask'])

# Converting to DataFrame
annotations_df = pd.DataFrame(annotations, columns=['filename', 'folder', 'class'])

# Saving to CSV
annotations_df.to_csv('annotations.csv', index=False)

# Load the annotations CSV
annotations_df = pd.read_csv('annotations.csv')

# Define the target image size
target_size = (224, 224)


# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image


# Load and preprocess images
images = []
labels = []

for index, row in annotations_df.iterrows():
    image_path = os.path.join(row['folder'], row['filename'])
    image = load_and_preprocess_image(image_path)

    label = row['class']
    label = 1 if label == 'mask' else 0

    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Converting to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images))

# Splitting the dataset into training and validation sets
dataset_size = len(images)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = dataset.take(train_size).batch(32)
val_dataset = dataset.skip(train_size).batch(32)


# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model


# Create and compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluating the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

# Saving the model
model.save('C:/Users/Ayush Thakur/Desktop/projects/New folder/saved_model.h5')

