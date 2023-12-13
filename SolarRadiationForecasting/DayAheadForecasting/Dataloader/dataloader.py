# This file will get dataset
# call all functions in data preprocessing
# split into train, val, test
# generate dataloaders

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from torch.utils.data.distributed import DistributedSampler

from Dataloader.PreprocessData import PreprocessData
from Dataloader.dataset import TiDEDataset

import os

# power_file = '/Datasets/weetwood_power.json'
# weather_file = '/Datasets/weetwood_weather.csv'

class TiDEDataloader():
    def __init__(self, args):

        self.batch_size = args.batch_size
        # self.power = os.path.join(args.root_path,
        #                                   args.power_path)
        # self.data_path = os.path.join(args.root_path,
        #                                   args.data_path)
        self.args = args


        self.get_data(args)

    def get_data(self, args):
        self.w_scaler = StandardScaler()
        self.t_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.pipeline = PreprocessData(args)
        # y, X_weather, X_time = self.pipeline.generate_clean_datasets()
        # df = generate_df('weetwood_power.json', 'weetwood_weather.csv')
        y_tr, X_weather_tr, X_time_tr = self.pipeline.generate_clean_datasets('train')
        y_cal, X_weather_cal, X_time_cal = self.pipeline.generate_clean_datasets('cal')
        y_val, X_weather_val, X_time_val = self.pipeline.generate_clean_datasets('val')
        y_te, X_weather_te, X_time_te = self.pipeline.generate_clean_datasets('test')

        # X_weather_tr, X_weather_cal, X_weather_val, X_weather_te = self.pipeline.split_X_data(X_weather)
        # X_time_tr, X_time_cal, X_time_val, X_time_te = self.pipeline.split_X_data(X_time)
        # y_tr, y_cal, y_val, y_te = self.pipeline.split_Y_data(y)

        # X_weather_tr, X_weather_val, X_weather_te = self.pipeline.split_X_data(X_weather)
        # X_time_tr, X_time_val, X_time_te = self.pipeline.split_X_data(X_time)
        # y_tr, y_val, y_te = self.pipeline.split_Y_data(y)

        y_tr = y_tr.reshape(-1, 1)
        y_cal = y_cal.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_te = y_te.reshape(-1, 1)

        X_hour_tr = X_time_tr[:,1]
        X_hour_cal = X_time_cal[:,1]
        X_hour_val = X_time_val[:,1]
        X_hour_te = X_time_te[:,1]

        self.w_scaler.fit(X_weather_tr)
        nXw_tr = self.w_scaler.transform(X_weather_tr)
        nXw_cal = self.w_scaler.transform(X_weather_cal)
        nXw_val = self.w_scaler.transform(X_weather_val)
        nXw_te = self.w_scaler.transform(X_weather_te)

        self.t_scaler.fit(X_time_tr)
        nXt_tr = self.t_scaler.transform(X_time_tr)
        nXt_cal = self.t_scaler.transform(X_time_cal)
        nXt_val = self.t_scaler.transform(X_time_val)
        nXt_te = self.t_scaler.transform(X_time_te)

        self.y_scaler.fit(y_tr)
        ny_tr = self.y_scaler.transform(y_tr)
        ny_cal = self.y_scaler.transform(y_cal)
        ny_val = self.y_scaler.transform(y_val)
        ny_te = self.y_scaler.transform(y_te)

        self.Yl_tr, self.Yh_tr = self.pipeline.generate_timeseries(ny_tr, self.args.step)
        self.Xw_tr = self.pipeline.generate_covariate_sequences(nXw_tr, self.args.step)
        self.Xt_tr = self.pipeline.generate_covariate_sequences(nXt_tr, self.args.step)
        self.Xh_tr = self.pipeline.generate_covariate_sequences(X_hour_tr, self.args.step)

        self.Yl_cal, self.Yh_cal = self.pipeline.generate_timeseries(ny_cal, self.args.step)
        self.Xw_cal = self.pipeline.generate_covariate_sequences(nXw_cal, self.args.step)
        self.Xt_cal = self.pipeline.generate_covariate_sequences(nXt_cal, self.args.step)
        self.Xh_cal = self.pipeline.generate_covariate_sequences(X_hour_cal, self.args.step)

        self.Yl_val, self.Yh_val = self.pipeline.generate_timeseries(ny_val, self.args.step)
        self.Xw_val = self.pipeline.generate_covariate_sequences(nXw_val, self.args.step)
        self.Xt_val = self.pipeline.generate_covariate_sequences(nXt_val, self.args.step)
        self.Xh_val = self.pipeline.generate_covariate_sequences(X_hour_val, self.args.step)

        self.Yl_te, self.Yh_te = self.pipeline.generate_timeseries(ny_te, self.args.step)
        self.Xw_te = self.pipeline.generate_covariate_sequences(nXw_te, self.args.step)
        self.Xt_te = self.pipeline.generate_covariate_sequences(nXt_te, self.args.step)     
        self.Xh_te = self.pipeline.generate_covariate_sequences(X_hour_te, self.args.step)


        # assume Xw, Xt, Yl, Yh are your numpy arrays


    def get_dataloader(self, flag):
        if flag == 'train':
            dataset = TiDEDataset(self.Xw_tr, self.Xt_tr, self.Xh_tr, self.Yl_tr, self.Yh_tr)
            if self.args.use_gpu and self.args.use_multi_gpu:
                sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, shuffle=False, drop_last=False)   
                dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
            else:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                        pin_memory=True, num_workers=self.args.num_workers) 
        elif flag == 'val':
            dataset = TiDEDataset(self.Xw_val, self.Xt_val, self.Xh_val, self.Yl_val, self.Yh_val)
            if self.args.use_gpu and self.args.use_multi_gpu:
                sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, shuffle=False, drop_last=False)   
                dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
            else:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                        pin_memory=True, num_workers=self.args.num_workers) 
        elif flag == 'cal':
            dataset = TiDEDataset(self.Xw_cal, self.Xt_cal, self.Xh_cal, self.Yl_cal, self.Yh_cal)
            if self.args.use_gpu and self.args.use_multi_gpu:
                sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, shuffle=False, drop_last=False)   
                dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
            else:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                        pin_memory=True, num_workers=self.args.num_workers) 
        else:
            dataset = TiDEDataset(self.Xw_te, self.Xt_te, self.Xh_te, self.Yl_te, self.Yh_te)
            if self.args.use_gpu and self.args.use_multi_gpu:
                sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, shuffle=False, drop_last=False)   
                dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
            else:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                        pin_memory=True, num_workers=self.args.num_workers) 

        return dataset, dataloader
