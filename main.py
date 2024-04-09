import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



Image_dir = '/train/images'

sample = 9
image_files = os.listdir(Image_dir)

# Randomly select num_samples images
rand_image = random.sample(image_files, sample)

fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i in range(sample):
    image = rand_image[i]
    ax = axes[i // 3, i % 3]
    ax.imshow(plt.imread(os.path.join(Image_dir, image)))
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()