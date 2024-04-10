import os
import random
import matplotlib.pyplot as plt

Image_dir = 'train/images'
Label_dir = 'train/labels'

# Define the classes
classes = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
           'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
           'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']

# Get a list of image files
image_files = os.listdir(Image_dir)

# Randomly select some images
num_samples = 9
rand_images = random.sample(image_files, num_samples)

# Plot the images along with their labels
fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i, image_file in enumerate(rand_images):
    image_path = os.path.join(Image_dir, image_file)
    label_path = os.path.join(Label_dir, image_file.replace('.jpg', '.txt'))

    # Read the image
    image = plt.imread(image_path)

    # Read the label
    with open(label_path, 'r') as file:
        label_data = file.readlines()

    # Plot the image
    ax = axes[i // 3, i % 3]
    ax.imshow(image)
    ax.set_title(f'Image {i+1}')

    # Add label information to the image
    label_str = ''
    for line in label_data:
        class_id, cx, cy, w, h = map(float, line.split())
        class_name = classes[int(class_id)]
        label_str += f'{class_name}: ({cx}, {cy}), w={w}, h={h}\n'

    ax.text(0, 1, label_str.strip(), fontsize=8, color='red', transform=ax.transAxes, verticalalignment='top')
    ax.axis('off')

plt.tight_layout()
plt.show()
