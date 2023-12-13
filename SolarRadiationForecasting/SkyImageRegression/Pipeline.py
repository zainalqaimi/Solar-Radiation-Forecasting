import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim


import datetime
from sklearn.preprocessing import StandardScaler

from Dataloader.ImageDataset import ImageDataset
from Dataloader.RegrDataset import RegrDataset
from metrics import metric
from Model.ResNet import ResNet
from Model.ResNetRegr import ResNetRegr
from Model.ViTRegression import ViTRegression
from tools import EarlyStopping, adjust_learning_rate

import time
import csv
import os

from torchvision import transforms

class Pipeline:
    def __init__(self, model_type, L, H, output_dim, hidden_dim, step, batch_size, lr, epochs, setting, 
                 blocks=16 , transform='normalise', device='cuda'):
        self.model_type = model_type
        self.L = L
        self.H = H
        self.step = step
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)

        self.blocks = blocks

        self.setting = setting

        self.mean = np.load('Dataloader/images_mean.npy')
        self.std = np.load('Dataloader/images_std.npy')

        self.transform_type = transform
        self.transform = self.create_transform(transform)

        # self.train_loader, self.cali_loader, self.vali_loader, self.test_loader = self.get_dataloader()

        self.model = self.build_model()

    def build_model(self):
        # model = ResNet(self.output_dim, self.hidden_dim, self.L, self.H).to(self.device)
        model = ResNetRegr(self.output_dim, self.blocks).to(self.device)
        # model = ViTRegression(self.output_dim).to(self.device)
        print("Forecasting model:")
        print(model)
        return model

    def create_transform(self, t_type):

        if t_type=='rescale':
            return transforms.Compose([
                transforms.Lambda(lambda x: x / 255.0)
            ])

        elif t_type=='normalise':
            return transforms.Compose([
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

        else:
            return transforms.Compose([
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        
    # def generate_sequences(self, data, dates):
    #     n = self.L + self.H
    #     Y = np.array([data[i:i+n] for i in range(len(data)-n + self.step)])
    #     Yl = Y[:, :self.L] # lookback/input sequence
    #     Yh = Y[:, self.L:] # horizon/target sequence
    #     return Yl, Yh


    def generate_sequences(self, data, dates):
        n = self.L + self.H
        Y = []  # List to store valid horizon/target sequences

        i = 0
        while i + n <= len(data):
        # Get the unique days in the current sequence
            unique_days = np.unique(dates[i:i+n])

            if len(unique_days) == 1:
                # All values in the current sequence have the same day, so add the sequence to Yl and Yh
                Y.append(data[i:i+n])

            i += self.step

        Y = np.array(Y)
        # Convert the lists to numpy arrays
        Yl = Y[:, :self.L] # lookback/input sequence
        Yh = Y[:, self.L:]

        return Yl, Yh

    def get_dataloader(self):

        if self.model_type == 'forecast':
            train_df = pd.read_csv('Dataloader/ForecastDatasets/fc_df_train.csv')
            # cal_df = pd.read_csv('Dataloader/folsom_images_cal.csv')
            val_df = pd.read_csv('Dataloader/ForecastDatasets/fc_df_val.csv')
            test_df = pd.read_csv('Dataloader/ForecastDatasets/fc_df_test.csv')

        else:
            train_df = pd.read_csv('Dataloader/RegressionDatasets/pre_df_train.csv')
            # cal_df = pd.read_csv('Dataloader/folsom_images_cal.csv')
            val_df = pd.read_csv('Dataloader/RegressionDatasets/pre_df_val.csv')
            test_df = pd.read_csv('Dataloader/RegressionDatasets/pre_df_test.csv')


        Y_tr = train_df['ghi'].to_numpy()
        # Y_cal = cal_df['ghi'].to_numpy()
        Y_val = val_df['ghi'].to_numpy()
        Y_te = test_df['ghi'].to_numpy()

        y_tr = Y_tr.reshape(-1, 1)
        # y_cal = Y_cal.reshape(-1, 1)
        y_val = Y_val.reshape(-1, 1)
        y_te = Y_te.reshape(-1, 1)

        X_img_tr = train_df['image_path'].to_numpy()
        # X_img_cal = cal_df['image_path'].to_numpy()
        X_img_val = val_df['image_path'].to_numpy()
        X_img_te = test_df['image_path'].to_numpy()

        time_tr_df = pd.DataFrame({
            'month': pd.to_datetime(train_df['timeStamp']).dt.month,
            'hour': pd.to_datetime(train_df['timeStamp']).dt.hour,
            'minute': pd.to_datetime(train_df['timeStamp']).dt.minute
        })
        date_tr = pd.to_datetime(train_df['timeStamp']).dt.date.to_numpy()

        # time_cal_df = pd.DataFrame({
        #     'month': pd.to_datetime(cal_df['timeStamp']).dt.month,
        #     'hour': pd.to_datetime(cal_df['timeStamp']).dt.hour,
        #     'minute': pd.to_datetime(cal_df['timeStamp']).dt.minute
        # })
        # date_cal = pd.to_datetime(cal_df['timeStamp']).dt.date.to_numpy()

        time_val_df = pd.DataFrame({
            'month': pd.to_datetime(val_df['timeStamp']).dt.month,
            'hour': pd.to_datetime(val_df['timeStamp']).dt.hour,
            'minute': pd.to_datetime(val_df['timeStamp']).dt.minute
        })
        date_val = pd.to_datetime(val_df['timeStamp']).dt.date.to_numpy()

        time_te_df = pd.DataFrame({
            'month': pd.to_datetime(test_df['timeStamp']).dt.month,
            'hour': pd.to_datetime(test_df['timeStamp']).dt.hour,
            'minute': pd.to_datetime(test_df['timeStamp']).dt.minute
        })
        date_test = pd.to_datetime(test_df['timeStamp']).dt.date.to_numpy()

        # Convert the DataFrame to a NumPy array
        Xt_tr = time_tr_df.to_numpy()
        # Xt_cal = time_cal_df.to_numpy()
        Xt_val = time_val_df.to_numpy()
        Xt_te = time_te_df.to_numpy()

        self.y_scaler = StandardScaler()
        self.t_scaler = StandardScaler()

        self.y_scaler.fit(y_tr)
        y_tr = self.y_scaler.transform(y_tr)
        # y_cal = self.y_scaler.transform(y_cal)
        y_val = self.y_scaler.transform(y_val)
        y_te = self.y_scaler.transform(y_te)

        self.t_scaler.fit(Xt_tr)
        Xt_tr = self.t_scaler.transform(Xt_tr)
        # Xt_cal = self.t_scaler.transform(Xt_cal)
        Xt_val = self.t_scaler.transform(Xt_val)
        Xt_te = self.t_scaler.transform(Xt_te)

        print(y_tr.shape, y_val.shape, y_te.shape)

        # Now need to generate sequences before passing through Dataset
        if self.model_type == 'forecast':
            Yl_tr, Yh_tr = self.generate_sequences(y_tr, date_tr)
            # Yl_cal, Yh_cal = self.generate_sequences(y_cal, date_cal)
            Yl_val, Yh_val = self.generate_sequences(y_val, date_val)
            Yl_te, Yh_te = self.generate_sequences(y_te, date_test)

            Xt_tr, _ = self.generate_sequences(Xt_tr, date_tr)
            # Xt_cal, _ = self.generate_sequences(Xt_cal, date_cal)
            Xt_val, _ = self.generate_sequences(Xt_val, date_val)
            Xt_te, _ = self.generate_sequences(Xt_te, date_test)

            X_img_tr, _ = self.generate_sequences(X_img_tr, date_tr)
            # X_img_cal, _ = self.generate_sequences(X_img_cal, date_cal)
            X_img_val, _ = self.generate_sequences(X_img_val, date_val)
            X_img_te, _ = self.generate_sequences(X_img_te, date_test)

            train_dataset = ImageDataset(Yl_tr, Yh_tr, X_img_tr, Xt_tr, self.mean, self.std, self.transform)
            # cal_dataset = ImageDataset(Yl_cal, Yh_cal, X_img_cal, Xt_cal, self.mean, self.std, self.transform)
            val_dataset = ImageDataset(Yl_val, Yh_val, X_img_val, Xt_val, self.mean, self.std, self.transform)
            test_dataset = ImageDataset(Yl_te, Yh_te, X_img_te, Xt_te, self.mean, self.std, self.transform)
        
        else:
            train_dataset = RegrDataset(y_tr, X_img_tr, Xt_tr, self.mean, self.std, self.transform)
            # cal_dataset = RegrDataset(y_cal, X_img_cal, Xt_cal, self.mean, self.std, self.transform)
            val_dataset = RegrDataset(y_val, X_img_val, Xt_val, self.mean, self.std, self.transform)
            test_dataset = RegrDataset(y_te, X_img_te, Xt_te, self.mean, self.std, self.transform)


        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)
        # cal_loader = DataLoader(cal_dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)
        te_loader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)

        # return train_loader, cal_loader, val_loader, te_loader
        return train_loader, val_loader, te_loader

    def train(self, train_loader, vali_loader, test_loader):
        scaler = torch.cuda.amp.GradScaler()
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(model_optim, self.lr, patience=5, verbose=True)
        
        path = os.path.join('./checkpoints/', self.setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        epoch_tr_loss = []
        epoch_val_loss = []
        epoch_te_loss = []

        for epoch in range(self.epochs):
            print("Epoch:", epoch)
            epoch_time = time.time()
            iter_count = 0
            train_loss = 0.
            self.model.train()
            # for i, ((batch_yl, batch_x_img, batch_xt), batch_yh) in enumerate(train_loader):
            for i, ((batch_x_img, batch_xt), batch_yh) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_img = batch_x_img.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                # batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)

                with torch.cuda.amp.autocast():
                    # outputs = self.model(batch_yl, batch_x_img, batch_xt)
                    outputs = self.model(batch_x_img, batch_xt)
                    batch_yh = batch_yh.to(self.device)

                    loss = criterion(outputs, batch_yh)


                # Check for nan values in model outputs
                # if torch.isnan(outputs).any():
                #     print("NaN detected in model outputs!")
                #     break

                # # Check for nan values in loss
                # if torch.isnan(loss).any():
                #     print("NaN detected in loss!")
                #     break

                train_loss += loss

                scaler.scale(loss).backward()

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print(f"NaN detected in gradients of {name}!")
                #         break


                scaler.step(model_optim)
                scaler.update()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss.item() / train_steps
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            epoch_tr_loss.append(train_loss)
            epoch_val_loss.append(vali_loss)
            epoch_te_loss.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(epoch+1, vali_loss, self.model, path)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # adjust_learning_rate(model_optim, epoch + 1, self.lr, 'type2')

        experiment_result = {
            'L': self.L,
            'H': self.H,
            'blocks': self.blocks,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'loss_function': 'mse',
            'batch_size': self.batch_size, 
            'learning_rate': self.lr, 
            'epochs': self.epochs, 
            'transform': self.transform_type,
            'train_loss_mse': train_loss, 
            'val_loss_mse': vali_loss, 
            'test_loss_mse': test_loss,
        }
        row = list(experiment_result.values())
        with open('img_experiments.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            # Write the result row
            writer.writerow(row)

        folder_path = './results/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'train_losses.npy', np.array(epoch_tr_loss))
        np.save(folder_path + 'val_losses.npy', np.array(epoch_val_loss))
        np.save(folder_path + 'test_losses.npy', np.array(epoch_te_loss))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return
        

    def vali(self, loader, criterion):
        total_loss = 0.
        self.model.eval()
        with torch.no_grad():
            # for i, ((batch_yl, batch_x_img, batch_xt), batch_yh) in enumerate(loader):
            for i, ((batch_x_img, batch_xt), batch_yh) in enumerate(loader):
                batch_x_img = batch_x_img.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                # batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)

                with torch.cuda.amp.autocast():
                    # outputs = self.model(batch_yl, batch_x_img, batch_xt)
                    outputs = self.model(batch_x_img, batch_xt)
                        
                batch_yh = batch_yh.to(self.device)

                pred = outputs
                true = batch_yh

                loss = criterion(pred, true)
                total_loss += loss

        total_loss = total_loss.item() / len(loader)
        # total_dtw_loss = total_dtw_loss.item() / len(vali_loader)
        self.model.train()
        return total_loss

    def test(self, test_loader, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.setting, 'checkpoint.pth')))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            # for i, ((batch_yl, batch_x_img, batch_xt), batch_yh) in enumerate(test_loader):
            for i, ((batch_x_img, batch_xt), batch_yh) in enumerate(test_loader):
                batch_x_img = batch_x_img.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                # batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)

                with torch.cuda.amp.autocast():
                    # outputs = self.model(batch_yl, batch_x_img, batch_xt)
                    outputs = self.model(batch_x_img, batch_xt)
                        
                outputs = outputs.detach().cpu().numpy()
                batch_yh = batch_yh.detach().cpu().numpy()

                pred = self.y_scaler.inverse_transform(outputs)
                true = self.y_scaler.inverse_transform(batch_yh)
                # pred = outputs
                # true = batch_yh

                if pred.shape[0] != self.batch_size or true.shape[0] != self.batch_size:
                    continue

                preds.append(pred)
                trues.append(true)

        folder_path = './results/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print(np.array(preds).shape, np.array(trues).shape)
        mae, mse, rmse, mape, mspe = metric(np.array(preds), np.array(trues))
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(folder_path + "result.txt", 'a')
        f.write(self.setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        # f.write('predicted:{}'.format(preds))
        f.write('\n')
        # f.write('true:{}'.format(trues))
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return