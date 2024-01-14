import shutil
import random
import os

# Define your dataset directory
dataset_dir = 'faces'

# Create directories for training and testing
train_dir = 'train_faces'
test_dir = 'test_faces'

# Define the ratio for splitting (e.g., 80% train, 20% test)
split_ratio = 0.8

for root, dirs, files in os.walk(dataset_dir):
    for directory in dirs:
        # Create corresponding train and test directories for each person
        os.makedirs(os.path.join(train_dir, directory), exist_ok=True)
        os.makedirs(os.path.join(test_dir, directory), exist_ok=True)

        images = os.listdir(os.path.join(dataset_dir, directory))
        random.shuffle(images)

        # Split images based on the defined ratio
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Move images to corresponding train and test folders
        for image in train_images:
            src = os.path.join(dataset_dir, directory, image)
            dest = os.path.join(train_dir, directory, image)
            shutil.copy(src, dest)

        for image in test_images:
            src = os.path.join(dataset_dir, directory, image)
            dest = os.path.join(test_dir, directory, image)
            shutil.copy(src, dest)

