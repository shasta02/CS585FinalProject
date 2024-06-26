import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Function to read and preprocess the images
def read_image(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image

# Function to read labels and bounding box data
def read_labels(label_path, num_classes=15):
    """
    Reads the labels from a given label file path.
    Returns the class label as a one-hot encoded vector and bounding box.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Initialize class_id and bbox with default values (no sign found)
    class_id = -1
    bbox = [0, 0, 0, 0]

    # If the file is not empty, process the lines
    if lines:
        class_id, cx, cy, w, h = map(float, lines[0].split())  # Assuming one object per image
        class_id = int(class_id)
        bbox = [cx, cy, w, h]

    # Handle the case where class_id is -1 (no sign)
    # It will create a one-hot vector of all zeros
    class_label_one_hot = to_categorical(class_id, num_classes=num_classes) if class_id != -1 else np.zeros(num_classes)
    
    return class_label_one_hot, np.array(bbox)


# Function to load the dataset and bounding box data
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

    images = np.array(images)
    class_labels = np.array(class_labels)
    bounding_boxes = np.array(bounding_boxes)

    return images, class_labels, bounding_boxes

# Load datasets
train_images, train_labels, train_bboxes = load_dataset('train', num_classes=15)
val_images, val_labels, val_bboxes = load_dataset('valid', num_classes=15)
test_images, test_labels, test_bboxes = load_dataset('test', num_classes=15)

# Define two inputs: one for the image and one for the bounding box data
image_input = Input(shape=(224, 224, 3), name='image_input')
bbox_input = Input(shape=(4,), name='bbox_input')

# Define the CNN for image processing
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)


# Define a simple feed-forward network for bounding box data
y = Dense(32, activation='relu')(bbox_input)
y = Dropout(0.5)(y)
y = Dense(64, activation='relu')(y)

# Concatenate the outputs of the two networks
combined = concatenate([x, y])

# Add a dense layer and the final classification layer
z = Dense(128, activation='relu')(combined)
output = Dense(15, activation='softmax', name='class_output')(z)

# Create the multi-input model
model = Model(inputs=[image_input, bbox_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Print model summary
print(model.summary())

# Train the model with both inputs
history = model.fit(
    [train_images, train_bboxes],  # Provide both inputs as a list
    train_labels,
    epochs=50,  # Adjust the number of epochs as needed
    validation_data=([val_images, val_bboxes], val_labels),
    batch_size=32
)
