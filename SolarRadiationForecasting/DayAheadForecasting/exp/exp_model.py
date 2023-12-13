from Dataloader.dataloader import TiDEDataloader
from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.loss.soft_dtw_cuda import SoftDTW

from TiDE.TiDE import TiDE
from MLP.MLP import MLP
# from utils.loss.dilate_loss import dilate_loss
# from tslearn.metrics import dtw, dtw_path
from CopulaCPTS.CopulaCPTS import CopulaCPTS

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time

import warnings
# import matplotlib.pyplot as plt
import numpy as np

import csv

# from ray.air import Checkpoint, session

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)
        self.args.device = self.device
        self.data = TiDEDataloader(self.args)
        self.model = self._build_model()
        self.sigma = 0.01
        self.gamma = 0.001
        self.alpha = 0.5

    def _build_model(self):
        # model = TiDE(self.args).to(self.device)
        # print(model)
        model = MLP(self.args).to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group("nccl", rank=self.args.rank, world_size=self.args.world_size)

            torch.cuda.set_device(self.args.rank)
            # print(self.args.x)
            # model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model = TiDE(self.args)
            model = DDP(model, device_ids=[self.args.rank], 
                                               output_device=self.args.rank)
            torch.cuda.set_device(self.args.rank)
            print("Built model:", model)

        return model

    def _get_data(self, flag):
        dataset, dataloader = self.data.get_dataloader(flag)
        return dataset, dataloader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # def setup(self, rank, world_size):
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def train(self, setting):
        # self.model = self._build_model()
        # self.setup(self.rank, self.world_size)
        train_data, train_loader = self._get_data(flag='train')
        # cali_data, cali_loader = self._get_data(flag='cal')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        epoch_tr_loss = []
        epoch_val_loss = []
        epoch_te_loss = []

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()

        # if self.args.loss == 'dtw':
        #     sdtw = SoftDTW(use_cuda=self.args.use_gpu, gamma=0.1)
        # else:
        #     sdtw = None

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # print(len(train_loader), self.args.train_epochs)
        for epoch in range(self.args.train_epochs):
            print("Epoch", epoch)
            iter_count = 0
            # train_loss = []
            train_loss = 0.

            self.model.train()
            epoch_time = time.time()

            if self.args.use_gpu and self.args.use_multi_gpu:
                print("Sampler")
                train_loader.sampler.set_epoch(epoch)

            for i, ((batch_xw, batch_xt, batch_yl, _), batch_yh) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_xw = batch_xw.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_yl, batch_xw, batch_xt)
                        # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))
                        batch_yh = batch_yh.to(self.device)

                        loss = criterion(outputs, batch_yh)
                        # if (self.args.loss=='mse'):
                        #     # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))
                        #     loss = criterion(outputs, batch_yh)                   
        
                        # elif (self.args.loss=='dilate'):   
                        #     loss, loss_shape, loss_temporal = dilate_loss(batch_yh,outputs,self.alpha, self.gamma, self.device)    
                            
                        # elif (self.args.loss=='dtw'):
                        #     # outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1)
                        #     loss = sdtw(outputs, batch_yh).mean()
                else:
                    outputs = self.model(batch_yl, batch_xw, batch_xt)
                    # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))
                    batch_yh = batch_yh.to(self.device)
            
                    loss = criterion(outputs, batch_yh)
                    # if (self.args.loss=='mse'):
                    #     # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))
                    #     loss = criterion(outputs, batch_yh)                  
        
                    # elif (self.args.loss=='dilate'):   
                    #     loss, loss_shape, loss_temporal = dilate_loss(batch_yh,outputs,self.alpha, self.gamma, self.device)    
                        
                    # elif (self.args.loss=='dtw'):
                    #     print("Calculating loss...")
                    #     # outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1)
                    #     loss = sdtw(outputs, batch_yh).mean()
                    

                    
                train_loss += loss

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(train_loss)
            train_loss = train_loss.item() / len(train_loader)
            vali_loss, vali_dtw_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_dtw_loss = self.vali(test_data, test_loader, criterion)

            epoch_tr_loss.append(train_loss)
            epoch_val_loss.append(vali_loss)
            epoch_te_loss.append(test_loss)

            # session.report(
            #     {"loss": vali_loss},
            # )

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} DTW Vali Loss: {5:.7f} DTW Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, vali_dtw_loss, test_dtw_loss))
            
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Your hyperparameters and loss values for the current experiment
        experiment_result = {
            'L': self.args.L,
            'H': self.args.H,
            'loss_function': self.args.loss,
            'batch_size': self.args.batch_size, 
            'learning_rate': self.args.learning_rate, 
            'epochs': self.args.train_epochs,
            'dropout': self.args.dropout, 
            'num_encoder_layers': self.args.num_enc_layers, 
            'num_decoder_layers': self.args.num_dec_layers, 
            'hidden_dim': self.args.hidden_dim, 
            'encoder_output_dim': self.args.enc_output_dim, 
            'decoder_output_dim': self.args.dec_output_dim, 
            'temp_hidden_dim': self.args.temporal_hidden_dim, 
            'train_loss_mse': train_loss, 
            'val_loss_mse': vali_loss, 
            'test_loss_mse': test_loss,
            'train_loss_dtw': None, 
            'val_loss_dtw': vali_dtw_loss, 
            'test_loss_dtw': test_dtw_loss
        }

        # Convert the result dictionary to a list
        row = list(experiment_result.values())

        # Open the file in append mode ('a') and create a writer object
        with open('experiments.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            # Write the result row
            writer.writerow(row)


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'train_losses.npy', np.array(epoch_tr_loss))
        np.save(folder_path + 'val_losses.npy', np.array(epoch_val_loss))
        np.save(folder_path + 'test_losses.npy', np.array(epoch_te_loss))


        # torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion, dtw=None):
        # total_loss = []
        total_loss = 0.
        total_dtw_loss = 0.
        self.model.eval()
        with torch.no_grad():
            for i, ((batch_xw, batch_xt, batch_yl, _), batch_yh) in enumerate(vali_loader):
                batch_xw = batch_xw.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)

                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_yl, batch_xw, batch_xt)
                        
                else:
                    outputs = self.model(batch_yl, batch_xw, batch_xt)
                    
                # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))
                batch_yh = batch_yh.to(self.device)

                pred = outputs
                true = batch_yh

                loss = criterion(pred, true)
                total_loss += loss

                # if dtw:
                #     # pred = pred.reshape(pred.size(0), pred.size(1), 1)
                #     # true = true.reshape(true.size(0), true.size(1), 1)
                #     dtw_loss = dtw(pred, true)
                #     total_dtw_loss += dtw_loss.mean()

        # if dtw:
        #     total_dtw_loss = total_dtw_loss.item() / len(vali_loader)
        # else:
        total_dtw_loss = 0

        total_loss = total_loss.item() / len(vali_loader)
        # total_dtw_loss = total_dtw_loss.item() / len(vali_loader)
        self.model.train()
        return total_loss, total_dtw_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        cal_data, cal_loader = self._get_data(flag='cal')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            #self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device('cpu')))

        preds = []
        trues = []

        Xw_cal = cal_data.Xw
        Xt_cal = cal_data.Xt
        Xh_cal = cal_data.Xh
        Yl_cal = cal_data.Yl

        Yh_cal = cal_data.Yh
 
        self.model.eval()
        self.copula = CopulaCPTS(self.args, self.model, Xw_cal, Xt_cal, Yl_cal, Yh_cal, Xh_cal)
        print("Calibrating copula...")
        self.copula.calibrate()
        eps = 0.1
        coverage = []
        
        print("Predicting copula...")
        box = self.copula.predict(epsilon=eps)
        area = self.copula.calc_area(box)

        print("testsize:", len(test_loader))
        with torch.no_grad():
            for i, ((batch_xw, batch_xt, batch_yl, batch_xh), batch_yh) in enumerate(test_loader):
                coverage_batch = []
                # print(i)
                batch_xw = batch_xw.float().to(self.device)
                batch_xt = batch_xt.float().to(self.device)
                batch_yl = batch_yl.float().to(self.device)

                batch_yh = batch_yh.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_yl, batch_xw, batch_xt)
                        # pred = torch.tensor(pred)
                        print("Calculating coverage...")
                        coverage_batch.append(self.copula.calc_coverage(box, outputs, batch_yh, batch_xh))
                else:
                    outputs = self.model(batch_yl, batch_xw, batch_xt)
                    coverage_batch.append(self.copula.calc_coverage(box, outputs, batch_yh, batch_xh))
                    
                # batch_yh = batch_yh.reshape(batch_yh.size(0), batch_yh.size(1))


                batch_yh = batch_yh.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_yh = batch_yh.detach().cpu().numpy()

                # pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                # true = batch_yh  # batch_y.detach().cpu().numpy()  # .squeeze()

                pred = self.data.y_scaler.inverse_transform(outputs)
                true = self.data.y_scaler.inverse_transform(batch_yh)

                if pred.shape[0] != self.args.batch_size or true.shape[0] != self.args.batch_size:
                    continue
                # print(pred.shape, true.shape)
                preds.append(pred)
                trues.append(true)

                coverage.append(coverage_batch)
        #         if i % 20 == 0:
        #             input = batch_x.detach().cpu().numpy()
        #             gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
        #             pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
        #             visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/Original_' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        coverage = np.array(coverage).mean(axis=0)


        mae, mse, rmse, mape, mspe = metric(np.array(preds), np.array(trues))
        print('mse:{}, mae:{}, coverage:{}'.format(mse, mae, coverage))
        f = open(folder_path + "result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{} coverage:{}'.format(mse, mae, coverage))
        f.write('\n')
        # f.write('predicted:{}'.format(preds))
        f.write('\n')
        # f.write('true:{}'.format(trues))
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'radius.npy', box)

        return
    
    def conformal(self):
        # copula = CopulaCPTS(model, X_cal, Y_cal)
        # copula.calibrate()

        # epsilon_ls = np.linspace(0.05, 0.50, 10)
        # area = []
        # coverage = []
        # for eps in epsilon_ls:
        #     pred, box = copula.predict(X_test, epsilon=eps)
        #     area.append(copula.calc_area(box))
        #     pred = torch.tensor(pred)
        #     coverage.append(copula.calc_coverage(box, pred, Y_test))

        return

    def predict(self, setting, load=False):
        
        return