# functions for preprocessing data
# to be called in dataloader
import pandas as pd
from datetime import datetime
from itertools import chain
import numpy as np
import torch

class PreprocessData():
    def __init__(self, args, data_path=None):
        # self.power = pd.read_json(power_file)
        # self.weather = pd.read_csv(weather_file)
        self.tr_df = pd.read_csv(args.train_path)
        self.cal_df = pd.read_csv(args.cal_path)
        self.val_df = pd.read_csv(args.val_path)
        self.te_df = pd.read_csv(args.test_path)
        # self.df = self.get_dataframe()

        self.L = args.L
        self.H = args.H
        self.step = args.step

        self.args = args

    def split_X_data(self, data):
        tr = int(0.6 * len(data))
        cal = int(0.2 * len(data))
        te = int(0.1 * len(data))

        train = data[0:tr]
        cali = data[tr:tr+cal]
        val = data[tr+cal:tr+cal+te]
        test = data[tr+cal+te:]

        return train, cali, val, test
        # tr = int(0.8 * len(data))
        # te = int(0.1 * len(data))

        train = data[0:tr]
        val = data[tr:tr+te]
        test = data[tr+te:]

        return train, val, test

    def split_Y_data(self, data):
        tr = int(0.6 * len(data))
        cal = int(0.2 * len(data))
        te = int(0.1 * len(data))

        train = data[0:tr]
        cali = data[tr:tr+cal]
        val = data[tr+cal:tr+cal+te]
        test = data[tr+cal+te:]

        return train, cali, val, test
    
        # tr = int(0.8 * len(data))
        # te = int(0.1 * len(data))

        # train = data[0:tr]
        # val = data[tr:tr+te]
        # test = data[tr+te:]

        # return train, val, test

    def handle_NaN(self):
        # self.df['rain_1h'] = self.df['rain_1h'].fillna(0)
        # self.df['snow_1h'] = self.df['snow_1h'].fillna(0)
        # self.df['visibility'] = self.df['visibility'].fillna(self.df['visibility'].groupby(self.df.date).transform('mean'))
        
        day1 = '2014-01-29'
        day2 = '2014-02-12'

        self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'])
        self.df = self.df[self.df['timeStamp'].dt.date != pd.to_datetime(day1).date()]
        self.df = self.df[self.df['timeStamp'].dt.date != pd.to_datetime(day2).date()]
        columns_to_fill = ['air_temp', 'relhum', 'windsp', 'winddir', 'precipitation']
        self.df[columns_to_fill] = self.df[columns_to_fill].fillna(method='ffill')

        # return data

    
    def generate_timeseries(self, data, step):
        n = self.L + self.H
        Y = np.array([data[i:i+n] for i in range(len(data)-n + step) if len(data[i:i+n]) == n])
        Yl = Y[:, :self.L] # lookback/input sequence
        Yh = Y[:, self.L:] # horizon/target sequence
        # Yl = Y[:self.L] # lookback/input sequence
        # Yh = Y[self.L:] # horizon/target sequence
        return Yl, Yh

    def generate_covariate_sequences(self, data, step):
        n = self.L + self.H
        sequences = np.array([data[i:i+n] for i in range(len(data)-n + step) if len(data[i:i+n]) == n])
        return sequences
    
    def generate_clean_datasets(self, flag):
        # self.handle_NaN()
        if flag=='train':
            df = self.tr_df
        elif flag=='cal':
            df = self.cal_df
        elif flag=='val':
            df = self.val_df
        elif flag=='test':
            df = self.te_df

        df = df.drop(['Unnamed: 0', 'timeStamp','time_int', 'date'], axis=1)

        w_variables = list(df.columns)

        to_remove = ['timeStamp', 'ghi', 'month', 'hour']
        w_variables = [i for i in w_variables if i not in to_remove]

        y = np.array(df.ghi)
        X_weather = np.array(df[w_variables])
        X_time = np.array(df[['month', 'hour']])
        # self.df = self.df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
        # self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        # self.df['Day'] = pd.to_datetime(self.df['Day'])
        
        # w_variables = list(self.df.columns)

        # to_remove = ['Global Solar Radiation (W/m2)', 'Timestamp', 'Day', 'month', 'hour']
        # w_variables = [i for i in w_variables if i not in to_remove]

        # y = np.array(self.df['Global Solar Radiation (W/m2)'])
        # X_weather = np.array(self.df[w_variables])
        # X_time = np.array(self.df[['month', 'hour']])

        # print(self.df.ghi.values)

        return y, X_weather, X_time
    

    # def normalise_X(self, data):
    #     # FIX AXIS
    #     min = np.min(data, keepdims=True)
    #     max = np.max(data, keepdims=True)
    #     norm = (data - min) / (max - min)

    #     return norm

    # def standardise_X(self, data):
    #     mean = np.mean(data, keepdims=True)
    #     std = np.std(data, keepdims=True)
    #     stand = (data - mean) / std

    #     return stand

    # def normalise_Y(self, data):
    #     min = np.min(data)
    #     max = np.max(data)
    #     norm = (data - min) / (max - min)

    #     return norm

    # def standardise_Y(self, data):
    #     mean = np.mean(data)
    #     std = np.std(data)
    #     stand = (data - mean) / std

    #     return stand
    
