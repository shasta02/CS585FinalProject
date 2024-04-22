import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def read_image(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = preprocess_input(image)  # Use preprocess_input from ResNet50
    return image

def read_labels(label_path, num_classes=15):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    class_id = -1
    bbox = [0, 0, 0, 0]

    if lines:
        class_id, cx, cy, w, h = map(float, lines[0].split())
        class_id = int(class_id)
        bbox = [cx, cy, w, h]

    class_label_one_hot = to_categorical(class_id, num_classes=num_classes) if class_id != -1 else np.zeros(num_classes)
    
    return class_label_one_hot, np.array(bbox)

def load_dataset(dataset_dir, num_classes=15):
    images = []
    class_labels = []
    bounding_boxes = []

    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        image = read_image(image_path)
        class_label, bbox = read_labels(label_path, num_classes)

        images.append(image)
        class_labels.append(class_label)
        bounding_boxes.append(bbox)

    return np.array(images), np.array(class_labels), np.array(bounding_boxes)

# Load datasets
train_images, train_labels, train_bboxes = load_dataset('train', num_classes=15)
val_images, val_labels, val_bboxes = load_dataset('valid', num_classes=15)
test_images, test_labels, test_bboxes = load_dataset('test', num_classes=15)

# Enhanced model with Batch Normalization and learning rate scheduling
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True  # Enable training in the last blocks

for layer in base_model.layers[:-10]:  # Freezing all layers except for the last 10
    layer.trainable = False

image_input = Input(shape=(224, 224, 3), name='image_input')
bbox_input = Input(shape=(4,), name='bbox_input')

x = base_model(image_input, training=True)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)

y = Dense(32, activation='relu')(bbox_input)
y = Dense(64, activation='relu')(y)

combined = concatenate([x, y])
output = Dense(15, activation='softmax', name='class_output')(combined)

model = Model(inputs=[image_input, bbox_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    [train_images, train_bboxes], train_labels,
    validation_data=([val_images, val_bboxes], val_labels),
    epochs=50, batch_size=32,
    callbacks=[reduce_lr, early_stopping]
)