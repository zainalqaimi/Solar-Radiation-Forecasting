import pandas as pd
from datetime import datetime
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuralforecast.auto import PatchTST
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MSE

def generate_clean_datasets(df):
  # Load the new dataset


  # Convert the 'timeStamp' column to datetime
  df['timeStamp'] = pd.to_datetime(df['timeStamp'])

  # Create the new dataframe with the desired structure
  df_transformed = df[['timeStamp', 'ghi']].copy()
  df_transformed.columns = ['ds', 'y']
  df_transformed['unique_id'] = 'Folsom'

  # Rearrange the columns to the desired order
  df_transformed = df_transformed[['unique_id', 'ds', 'y']]

  return df_transformed


tr_df = pd.read_csv('./Datasets/folsom_train.csv')
cal_df = pd.read_csv('./Datasets/folsom_cal.csv')
val_df = pd.read_csv('./Datasets/folsom_val.csv')
te_df = pd.read_csv('./Datasets/folsom_test.csv')

# df = generate_df('weetwood_power.json', 'weetwood_weather.csv')
tr_df = generate_clean_datasets(tr_df)
cal_df = generate_clean_datasets(cal_df)
val_df = generate_clean_datasets(val_df)
te_df = generate_clean_datasets(te_df)

model = PatchTST(h=96,
                 input_size=192,
                 patch_len=16,
                 stride=4,
                 revin=False,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=MSE(),
                 #loss=MAE(),
                 learning_rate=1e-3,
                 max_steps=100)

nf = NeuralForecast(
    models=[model],
    freq='0h15min'
)

nf.fit(df=tr_df)