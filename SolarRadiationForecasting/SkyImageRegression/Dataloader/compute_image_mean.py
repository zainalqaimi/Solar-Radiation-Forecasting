import numpy as np
import pandas as pd
from torchvision import transforms
import os
import cv2

BATCH_SIZE = 64

def calculate_mean_std(image_paths):
    cumulative_sum = np.zeros(3)  # Assuming images are RGB, so 3 channels
    count = 0

    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        # print(batch_start)
        batch_end = min(batch_start + BATCH_SIZE, len(image_paths))
        batch_image_paths = image_paths[batch_start:batch_end]
        batch_cumulative_sum = np.zeros(3)
        batch_count = 0

        for path in batch_image_paths:
            path = os.path.join('../../', path)
            img = cv2.imread(path)
            img = img / 255.0  # Normalize pixel values to [0, 1]
            batch_cumulative_sum += np.sum(img, axis=(0, 1))
            batch_count += img.shape[0] * img.shape[1]

        cumulative_sum += batch_cumulative_sum
        count += batch_count

    mean = cumulative_sum / count

    # Compute the standard deviation using matrix operations
    squared_cumulative_sum = np.zeros(3)
    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        # print(batch_start)
        batch_end = min(batch_start + BATCH_SIZE, len(image_paths))
        batch_image_paths = image_paths[batch_start:batch_end]
        batch_squared_cumulative_sum = np.zeros(3)

        for path in batch_image_paths:
            path = os.path.join('../../', path)
            img = cv2.imread(path)
            img = img / 255.0  # Normalize pixel values to [0, 1]
            batch_squared_cumulative_sum += np.sum((img - mean) ** 2, axis=(0, 1))

        squared_cumulative_sum += batch_squared_cumulative_sum

    std = np.sqrt(squared_cumulative_sum / count)

    return mean, std

def normalize_image_batch(images, mean, std):
    # Normalize all images in the batch using matrix operations
    normalized_images = (images - mean) / std
    return normalized_images

def initialize_normalized_column(df):
    # Initialize the 'normalized_image' column with empty lists
    df['normalized_image'] = [[] for _ in range(len(df))]
    return df

def normalize_dataframe(df, mean, std):
    num_samples = len(df)
    df = initialize_normalized_column(df)

    for batch_start in range(0, num_samples, BATCH_SIZE):
        print(batch_start)
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        batch_images = np.stack(df['image'].iloc[batch_start:batch_end].values)
        normalized_batch_images = normalize_image_batch(batch_images, mean, std)
        df['normalized_image'].iloc[batch_start:batch_end] = normalized_batch_images.tolist()

    return df

# Assume df_train is the DataFrame containing training data with columns 'timeStamp', 'ghi', and 'image_path'
df_train = pd.read_csv('folsom_images_train.csv')
# df_cal = pd.read_csv('folsom_images_cal.csv')
# df_val = pd.read_csv('folsom_images_val.csv')
# df_test = pd.read_csv('folsom_images_test.csv')

print("Train size:", len(df_train))
# print("Cal size:", len(df_cal))
# print("Val size:", len(df_val))
# print("Test size:", len(df_test))

train_image_paths = df_train['image_path'].tolist()
mean, std = calculate_mean_std(train_image_paths)
print("Mean:", mean)
print("STD:", std)

# Save the mean and std to a file for later use during testing
np.save("images_mean.npy", mean)
np.save("images_std.npy", std)

# Load the mean and std if needed later
# mean = np.load("images_mean.npy")
# std = np.load("images_std.npy")

# Normalize the training set and save it as DataFrame
# train_image_paths = train_images
# train_images = [cv2.imread(path) for path in train_image_paths]
# df_train['image'] = train_images
# # df_train_normalized = normalize_dataframe(df_train.copy(), mean, std)
# df_train.to_csv("train_full_images.csv", index=False)
# print(df_train.head(5))

# cal_image_paths = df_cal['image_path'].apply(lambda x: os.path.join('../Datasets/', x)).tolist()
# cal_images = [cv2.imread(path) for path in cal_image_paths]
# df_cal['image'] = cal_images
# # df_cal_normalized = normalize_dataframe(df_cal.copy(), mean, std)
# df_cal.to_csv("cal_full_images.csv", index=False)
# print(df_cal.head(5))
# print(len(df_cal))


# # Assume df_val is the DataFrame containing validation data with columns 'timeStamp', 'ghi', and 'image_path'
# val_image_paths = df_val['image_path'].apply(lambda x: os.path.join('../Datasets/', x)).tolist()
# val_images = [cv2.imread(path) for path in val_image_paths]
# df_val['image'] = val_images
# # df_val_normalized = normalize_dataframe(df_val.copy(), mean, std)
# df_val.to_csv("val_full_images.csv", index=False)
# print(df_val.head(5))
# print(len(df_val))


# # Assume df_test is the DataFrame containing test data with columns 'timeStamp', 'ghi', and 'image_path'
# test_image_paths = df_test['image_path'].apply(lambda x: os.path.join('../Datasets/', x)).tolist()
# test_images = [cv2.imread(path) for path in test_image_paths]
# df_test['image'] = test_images
# # df_test_normalized = normalize_dataframe(df_test.copy(), mean, std)
# df_test.to_csv("test_full_images.csv", index=False)
# print(df_test.head(5))
# print(len(df_test))


