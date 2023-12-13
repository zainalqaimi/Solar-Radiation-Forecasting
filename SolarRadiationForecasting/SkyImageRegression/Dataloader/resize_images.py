import cv2
import os
import pandas as pd

# Assuming you have the image paths in a list called 'image_paths'
df = pd.read_csv('folsom_image_dataset.csv')
image_paths = df['image_path']

# Specify the target size for resizing
target_size = (224, 224)

print("Resizing...")
# Loop over all images and resize them
for path in image_paths:
    # print(path)
    # Read the image using OpenCV
    path = os.path.join('../Datasets/', path)
    img = cv2.imread(path)

    current_size = (img.shape[1], img.shape[0])

    # Check if the current size is equal to the target size
    if current_size != target_size:
        print(path)

    # Resize the image to the target size
    # resized_img = cv2.resize(img, target_size)

    # Overwrite the original image file with the resized image
    # cv2.imwrite(path, resized_img)

# After this loop, the images in 'image_paths' will be resized and overwritten
