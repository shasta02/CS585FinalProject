import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models

Image_dir = 'train/images'

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

# Load datasets
train_images, train_labels, train_boxes = load_dataset('train', num_classes=15)
val_images, val_labels, val_boxes = load_dataset('valid', num_classes=15)
test_images, test_labels, test_boxes = load_dataset('test', num_classes=15)

# Model definition
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

# Compile the model
model.compile(optimizer='adam',
              loss='mse',  # Mean Squared Error for bounding box and possibly classification
              metrics=['accuracy'])

# Print model summary
print(model.summary())

def preprocess_labels(labels, boxes, num_classes=15):
    num_samples = labels.shape[0]
    processed_labels = np.zeros((num_samples, num_classes + 4), dtype=np.float32)
    
    for idx, (label, box) in enumerate(zip(labels.to_list(), boxes.to_list())):
        # Flatten and reshape the one-hot encoded class labels
        one_hot_label = tf.keras.utils.to_categorical(label, num_classes=num_classes).flatten()
        
        one_hot_label = one_hot_label.reshape((num_classes, num_classes))
        # Concatenate the flattened one-hot encoded class label with the bounding box
        processed_labels[idx, :num_classes] = one_hot_label
        processed_labels[idx, num_classes:] = box
    
    return processed_labels

# Preprocess labels for training and validation
train_targets = preprocess_labels(train_labels, train_boxes, num_classes=15)
val_targets = preprocess_labels(val_labels, val_boxes, num_classes=15)

# Train the model
history = model.fit(
    train_images, 
    train_targets, 
    epochs=1,
    validation_data=(val_images, val_targets),
    batch_size=32
)
