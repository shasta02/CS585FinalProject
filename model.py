import os
import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image

from keras import layers, models

# from sklearn.model_selection import train_test_split 
# from keras.utils import to_categorical 
# from keras.models import Sequential 
# from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



Image_dir = 'train/images'

# sample = 9
# image_files = os.listdir(Image_dir)

# # Randomly select num_samples images
# rand_image = random.sample(image_files, sample)

# fig, axes = plt.subplots(3, 3, figsize=(11, 11))

# for i in range(sample):
#     image = rand_image[i]
#     ax = axes[i // 3, i % 3]
#     ax.imshow(plt.imread(os.path.join(Image_dir, image)))
#     ax.set_title(f'Image {i+1}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()


# image = cv2.imread("train/images/00014_00006_00011_png.rf.7c2d9a379350594748c581895b5c5ea1.jpg")
# h, w, c = image.shape
# plt.imshow(image)
# plt.title(f"Image Shape: {w} x {h} and 3 channels")
# plt.show()


def read_image(image_path, image_size=(224, 224)):
    """
    Reads an image from a given path and resizes it to the given size.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def read_labels(label_path, num_classes=15):
    """
    Reads the labels from a given label file path.
    Returns a list of class labels and bounding boxes.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    classes = []
    boxes = []
    for line in lines:
        class_id, cx, cy, w, h = map(float, line.split())
        classes.append(int(class_id))
        boxes.append([cx, cy, w, h])  # You might need to adjust this depending on how you want to use the bounding boxes
    
    # Convert class labels to one-hot encoded vectors
    classes_one_hot = tf.keras.utils.to_categorical(classes, num_classes=num_classes)
    
    return classes_one_hot, np.array(boxes)


def load_dataset(dataset_dir, num_classes=15):
    """
    Loads the dataset from a given directory.
    Returns lists of images, class labels as RaggedTensors, and bounding boxes as RaggedTensors.
    """
    images = []
    class_labels = []
    bounding_boxes = []
    
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        image = read_image(image_path)
        classes_one_hot, boxes = read_labels(label_path, num_classes)
        
        images.append(image)
        class_labels.append(classes_one_hot)
        bounding_boxes.append(boxes)
    
    return (np.array(images), 
            tf.ragged.constant(class_labels, dtype=tf.float32), 
            tf.ragged.constant(bounding_boxes, dtype=tf.float32))

train_images, train_labels, train_boxes = load_dataset('train', num_classes=15)

# val_images, val_labels, val_boxes = load_dataset('path/to/validation', num_classes=15)
# test_images, test_labels, test_boxes = load_dataset('path/to/test', num_classes=15)

classes = 15
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # Output layer: number of classes + 4 for bounding box (x, y, w, h)
    layers.Dense(classes + 4)
])

model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'box_output': 'mse'},
              metrics=['accuracy'])

print(model.summary())