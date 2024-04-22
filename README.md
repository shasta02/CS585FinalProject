# Traffic Sign Recognition Models Comparison

This README details the differences between two traffic sign recognition models developed using TensorFlow and Keras. The first model focuses on basic image and bounding box processing, while the second utilizes transfer learning with the pre-trained ResNet50 model for enhanced feature extraction.

## Model 1: Basic Traffic Sign Recognition Model

### Overview
- **Image Processing**: Implements a series of convolutional and pooling layers to process images.
- **Bounding Box Processing**: Handles bounding box data through a sequence of fully connected layers.
- **Feature Integration**: Features from image and bounding box processing are merged prior to classification.

### Key Features
- Separate processing stream for bounding box data using dense layers.
- Combined processing of image and bounding box data for final output.

&nbsp;





## Model 2: Advanced Traffic Sign Recognition Using Transfer Learning

### Overview
- Incorporates the ResNet50 model, a pre-trained network on the ImageNet dataset, to enhance initial feature detection and extraction.

### Key Features
- **Pre-trained Network**: Uses ResNet50 with its lower layers set to non-trainable to preserve learned features.
- **Optimization Techniques**: Includes dropout for regularization and a lower learning rate (0.0001) specifically for fine-tuning.
- **Preprocessing**: Employs `preprocess_input` from ResNet50 to ensure the input data is suitably normalized and scaled.

&nbsp;

