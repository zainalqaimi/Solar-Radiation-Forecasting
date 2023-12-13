import pandas as pd
from datetime import datetime, timedelta
from itertools import chain
import numpy as np
import os

# Need to generate array for image paths, and corresponding GHI value

root = "../Datasets/"
ghi_file = "folsom_dataset_15m.csv"
images_file = "SkyImages2015/2015_15m"

ghi_path = os.path.join(root, ghi_file)
images_path = os.path.join(root, images_file)

df = pd.read_csv(ghi_path)
# get ghi, timestamp, month, hour
columns = ['timeStamp', 'ghi', 'month', 'hour']
ghi = df[columns]
ghi['timeStamp'] = pd.to_datetime(ghi['timeStamp'])

# Define the exact minutes we're interested in
exact_minutes = {0, 15, 30, 45}
# Define the minutes where we consider 59 seconds before the next minute
previous_minutes = {14, 29, 44, 59}

image_paths = []
image_dts = []
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):  # or whatever file types you're using
        dt_str = filename[:-4]  # remove the .jpg
        dt = datetime.strptime(dt_str, '%Y%m%d_%H%M%S')
        image_paths.append(filename)
        if dt.minute in previous_minutes and dt.second == 59:
            dt = dt + timedelta(seconds=1)
        image_dts.append(dt)


ghi = ghi[ghi['timeStamp'].isin(image_dts)]

df_timeStamps = set(ghi['timeStamp'])

# Convert image_dts to a set
image_dts_set = set(image_dts)

# Find the timestamps in image_dts but not in df['timeStamp']
missing_timestamps = image_dts_set - df_timeStamps

# Use list comprehension to get new lists without the missing timestamps
# Pair each image path with its corresponding datetime
paired = list(zip(image_dts, image_paths))

# Sort the pairs based on the datetime
paired.sort(key=lambda x: x[0])

# Unzip the pairs
image_dts, image_paths = zip(*paired)

# Now, both lists are sorted by datetime and we can proceed to remove elements

# use list comprehension to filter out the image_dts and image_paths
image_dts = [dt for dt in image_dts if dt not in missing_timestamps]
image_paths = [path for dt, path in zip(image_dts, image_paths) if dt not in missing_timestamps]

# print(ghi.head(50))
# print(image_dts[:50])
# print(image_paths[:50])
# print(len(ghi), len(image_dts), len(image_paths))

ghi['image_path'] = np.array(image_paths)
df = ghi

# split into train/test/val


df.to_csv("image_dataset.csv")
