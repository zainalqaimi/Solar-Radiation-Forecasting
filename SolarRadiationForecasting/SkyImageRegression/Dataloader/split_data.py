import pandas as pd

# Load the dataframe
df = pd.read_csv('../Datasets/image_dataset.csv')

# Calculate the sizes of the train, validation and test datasets
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

# Split the data
train = df[:train_size]
val = df[train_size:(train_size+val_size)]
test = df[(train_size+val_size):]

# Save these dataframes as separate csv files
train.to_csv('../Datasets/images_train.csv', index=False)
val.to_csv('../Datasets/images_val.csv', index=False)
test.to_csv('../Datasets/images_test.csv', index=False)
