import pandas as pd
from datetime import datetime, timedelta
from itertools import chain
import numpy as np
import os
from collections import Counter

import pytz

# Need to generate array for image paths, and corresponding GHI value
def generate_image_path(filename):
    # Extract the year, month, and day from the filename
    # year = filename[:4]
    # month = filename[4:6]
    # day = filename[6:8]

    # Create the path components
    path_components = ['new_folsom_images']

    # Join the path components and append the filename
    image_path = os.path.join(*path_components, filename)

    return image_path

root = "../Datasets/"
ghi_file = "Folsom_ghi.csv"
images_folder = "new_folsom_images/"

ghi_path = os.path.join(root, ghi_file)
images_path = os.path.join(root, images_folder)

df = pd.read_csv(ghi_path)
# get ghi, timestamp, month, hour
columns = ['timeStamp', 'ghi']
ghi = df[columns]
ghi['timeStamp'] = pd.to_datetime(ghi['timeStamp'])

# utc_tz = pytz.timezone('UTC')
# la_tz = pytz.timezone('America/Los_Angeles')

# Define the exact minutes we're interested in
# We want to round each file to the nearest 1/4 of an hour
# exact_minutes = {0, 15, 30, 45}

# Define the minutes where we consider 59 seconds before the next minute
# previous_minutes = {14, 29, 44, 59}

image_paths = []
image_dts = []
image_dict = {}
# for filename in os.listdir(images_path):
    # for month in year
    # for day in month
    # 'Sky/Filename'
    # 
# Walk through the root directory
import os

# Specify the directory you want to sort
# Get a sorted list of all file names in the directory
print("Sorting files...")
sorted_files = sorted(os.listdir(images_path))

print("Starting iteration...")
for filename in sorted_files:
    if filename.endswith(".jpg"):  # or whatever file types you're using
        dt_str = filename.split('_')  # get the date and time string
        dt_str = (''.join(dt_str))[:-4]

        dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
        dt_rounded = pd.Timestamp(dt).floor('min').to_pydatetime()
        if dt_rounded in image_dict.keys():
            dt_rounded = pd.Timestamp(dt).ceil('min').to_pydatetime()

        new_filename = dt_rounded.strftime('%Y%m%d_%H%M%S.jpg')

        filepath = generate_image_path(new_filename)

        original_file_path = os.path.join(images_path, filename)
        new_file_path = os.path.join(images_path, new_filename)
        os.rename(original_file_path, new_file_path)

        image_paths.append(filepath)
        image_dts.append(dt_rounded)
        if dt_rounded in image_dict.keys():
            image_dict[dt_rounded].append(dt)
        else:
            image_dict[dt_rounded] = [dt]


print(len(image_dts), len(image_paths))
ghi = ghi[ghi['timeStamp'].isin(image_dts)]

df_timeStamps = set(ghi['timeStamp'])
image_dts_set = set(image_dts)
print(len(ghi), len(df_timeStamps))


# Count the occurrences of each value in the array
counter = Counter(image_dts)

# Find the values that occur more than once
duplicates = [value for value, count in counter.items() if count > 1]

print("Duplicates:", len(duplicates))
# print(duplicates[0:10])
# print("Dictionary:", image_dict[duplicates[0]])


# Convert image_dts to a set

print(len(image_dts_set), len(df_timeStamps))
missing_timestamps = image_dts_set - df_timeStamps
print("Num missing:", len(missing_timestamps))
# print("Missing:", sorted(list(missing_timestamps))[:100])


# Use list comprehension to get new lists without the missing timestamps
# Pair each image path with its corresponding datetime
paired = list(zip(image_dts, image_paths))

# Sort the pairs based on the datetime
paired.sort(key=lambda x: x[0])

# Unzip the pairs
image_dts, image_paths = zip(*paired)
print("dts:", image_dts[:10])
print("paths:", image_paths[:10])

# Now, both lists are sorted by datetime and we can proceed to remove elements

# use list comprehension to filter out the image_dts and image_paths

# image_dts = [dt for dt in image_dts if dt not in missing_timestamps]
image_paths = [path for dt, path in zip(image_dts, image_paths) if dt not in missing_timestamps]

# print(ghi.head(50))
# print(image_dts[:50])
# print(image_paths[:50])
print(len(ghi), len(image_dts), len(image_paths))

ghi['image_path'] = np.array(image_paths)
df = ghi

# split into train/test/val

print(len(df))
print(df.head(10))
df.to_csv("folsom_image_dataset.csv")
