# This file is the main script
# to train and test all models

import numpy as np
import argparse
import os
import torch
import torch.multiprocessing as mp
from exp.exp_model import Exp_Model
import random



def main():
    fix_seed = 7
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TiDE')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model', type=str, required=False, default='TiDE', help='model name')
                        # help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    # parser.add_argument('--root_path', type=str, default='./Datasets/', help='root path of the data file')
    parser.add_argument('--train_path', type=str, default='./Datasets/folsom_train.csv', help='data file')
    parser.add_argument('--cal_path', type=str, default='./Datasets/folsom_cal.csv', help='data file')
    parser.add_argument('--val_path', type=str, default='./Datasets/folsom_val.csv', help='data file')
    parser.add_argument('--test_path', type=str, default='./Datasets/folsom_test.csv', help='data file')
    # parser.add_argument('--freq', type=str, default='h',
                        # help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # forecasting task
    parser.add_argument('--L', type=int, default=192, help='input sequence length')
    parser.add_argument('--H', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--step', type=int, default=1, help='Step size for segmenting time series')

    # model define
    parser.add_argument('--proj_input_dim', type=int, default=5, help='feature projection input dimension')
    parser.add_argument('--proj_hidden_dim', type=int, default=4, help='feature projection hidden dimension')
    parser.add_argument('--proj_output_dim', type=int, default=4, help='feature projection output dimension')
    parser.add_argument('--t2v_input_dim', type=int, default=2, help='t2v input dimension')
    parser.add_argument('--t2v_output_dim', type=int, default=2, help='t2v output dimension')
    parser.add_argument('--num_enc_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--num_dec_layers', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='encoder/decoder hidden dimensions')
    parser.add_argument('--enc_output_dim', type=int, default=32, help='encoder output dimebsion/decoder input dimension')
    parser.add_argument('--dec_output_dim', type=int, default=64, help='decoder output dimension, to be multiplied by H')

    parser.add_argument('--temporal_hidden_dim', type=int, default=512, help='temporal decoder hidden dimension')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--do_predict', default=False, action='store_true', help='whether to predict unseen future data')
    
    args = parser.parse_args()
    # enc_input_dim = args.L + ((args.L+args.H)*(args.t2v_output_dim))
    enc_input_dim = args.L + ((args.L+args.H)*(args.proj_output_dim+args.t2v_output_dim))
    final_dec_output_dim = args.H * args.dec_output_dim

    parser.add_argument('--enc_input_dim', type=int, default=enc_input_dim, help='number of decoder layers')

    # parser.add_argument('--enc_input_dims', nargs='+', type=int, default=[enc_input_dim, 128])
    parser.add_argument('--enc_input_dims', nargs='+', type=int, default=[enc_input_dim]+[args.hidden_dim]*(args.num_enc_layers-1))

    # parser.add_argument('--enc_hidden_dims', nargs='+', type=int, default=[256,64])
    parser.add_argument('--enc_hidden_dims', nargs='+', type=int, default=[args.hidden_dim]*args.num_enc_layers)

    # parser.add_argument('--enc_out_dims', nargs='+', type=int, default=[128,32])
    parser.add_argument('--enc_out_dims', nargs='+', type=int, default=[args.hidden_dim]*(args.num_enc_layers-1)+[args.enc_output_dim])

    # parser.add_argument('--dec_input_dims', nargs='+', type=int, default=[32,128])
    parser.add_argument('--dec_input_dims', nargs='+', type=int, default=[args.enc_output_dim]+[args.hidden_dim]*(args.num_dec_layers-1))

    # parser.add_argument('--dec_hidden_dims', nargs='+', type=int, default=[64,256])
    parser.add_argument('--dec_hidden_dims', nargs='+', type=int, default=[args.hidden_dim]*args.num_dec_layers)

    # parser.add_argument('--dec_out_dims', nargs='+', type=int, default=[128,final_dec_output_dim])
    parser.add_argument('--dec_out_dims', nargs='+', type=int, default=[args.hidden_dim]*(args.num_dec_layers-1)+[final_dec_output_dim])


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        args.world_size = len(device_ids)
        args.rank = args.device_ids[0]
    if args.use_gpu and not args.use_multi_gpu:
        # args.devices = args.devices.replace(' ', '')
        # device_ids = args.devices.split(',')
        print(args.gpu)
        args.world_size = 1
        args.rank = args.gpu

    print('Args in experiment:')
    print(args)

    Exp = Exp_Model

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_L{}_H{}_encl{}_decl{}_hdim{}_eo{}_do{}_th{}_drop{}_epochs{}_bs{}_lr{}_loss{}'.format(
                args.model,
                args.L,
                args.H,
                args.num_enc_layers,
                args.num_dec_layers,
                args.hidden_dim,
                args.enc_output_dim,
                args.dec_output_dim,
                args.temporal_hidden_dim,
                args.dropout,
                args.train_epochs,
                args.batch_size,
                args.learning_rate,
                args.loss)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

######### 
            if args.use_gpu and args.use_multi_gpu:
                mp.spawn(
                    exp.train,
                    args=(setting),
                    nprocs=args.world_size)
            else:
                exp.train(setting)
#########

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_L{}_H{}_encl{}_decl{}_hdim{}_eo{}_do{}_th{}_drop{}_epochs{}_bs{}_lr{}_loss{}'.format(
            args.model,
            args.L,
            args.H,
            args.num_enc_layers,
            args.num_dec_layers,
            args.hidden_dim,
            args.enc_output_dim,
            args.dec_output_dim,
            args.temporal_hidden_dim,
            args.dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.loss)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()