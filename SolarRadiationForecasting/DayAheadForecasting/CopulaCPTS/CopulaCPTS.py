import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from copulae import GumbelCopula
# from copulae.core import pseudo_obs
# from .utils import search_alpha
from CopulaCPTS.CP import CP, search_alpha

from scipy.stats import rankdata

import math

def pseudo_obs(data):
    ranks = rankdata(data)  # Compute ranks of the data points, handling ties
    pseudo_data = ranks / (len(data) + 1)  # Convert ranks to pseudo-observations

    return pseudo_data


def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[0])))
                           ) - 1 + epsilon)

class CopulaCPTS:
    '''
    copula based conformal prediction time series
    '''
    def __init__(self, args, model, Xw_cal, Xt_cal, Yl_cal, Yh_cal, Xh_cal):
        """
        har har
        """
        self.args = args
        self.model = model

        self.cali_xw = None
        self.cali_xt = None
        self.cali_yl = None
        self.cali_yh = None
        self.cali_xh = None
        self.copula_xw = None
        self.copula_xt = None
        self.copula_yl = None
        self.copula_yh = None
        self.copula_xh = None
        self.split_cali(Xw_cal, Xt_cal, Yl_cal, Yh_cal, Xh_cal)
        
        self.nonconformity = None
        self.results_dict = {}

    def split_cali(self, Xw_cal, Xt_cal, Yl_cal, Yh_cal, Xh_cal, split=0.6):
        if self.copula_xw:
            print("already split")
            return 
        size = Xw_cal.shape[0]
        halfsize = int(split*size)
        
        idx = np.random.choice(range(size), halfsize, replace=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cali_xw = torch.from_numpy(Xw_cal[idx]).float().to(device)
        self.cali_xt = torch.from_numpy(Xt_cal[idx]).float().to(device)
        self.cali_yl = torch.from_numpy(Yl_cal[idx]).float().to(device)
        self.cali_xh = torch.from_numpy(Xh_cal[idx]).float().to(device)

        self.copula_xw = torch.from_numpy(Xw_cal[list(set(range(size)) - set(idx))]).float().to(device)
        self.copula_xt = torch.from_numpy(Xt_cal[list(set(range(size)) - set(idx))]).float().to(device)
        self.copula_yl = torch.from_numpy(Yl_cal[list(set(range(size)) - set(idx))]).float().to(device)
        self.copula_xh = torch.from_numpy(Xh_cal[list(set(range(size)) - set(idx))]).float().to(device)
        
        self.cali_yh = torch.from_numpy(Yh_cal[idx]).float().to(device)
        self.copula_yh = torch.from_numpy(Yh_cal[list(set(range(size)) - set(idx))]).float().to(device)



    def calibrate(self):
        ###################### ORIGINAL CODE ##############################
        pred_y = self.model(self.cali_yl, self.cali_xw, self.cali_xt)
        nonconformity = torch.norm((pred_y-self.cali_yh), p=2, dim = -1)
        self.nonconformity = nonconformity.detach().numpy()
        ###################################################################

        # Initialize a list of 24 empty lists to store nonconformity scores
        
        # Generate predictions and calculate nonconformity scores
        # pred_y = self.model(self.cali_yl, self.cali_xw, self.cali_xt)
        # scores = torch.norm((pred_y-self.cali_yh), p=2, dim = -1).cpu().detach().numpy()
        # print(scores.shape)
        # self.nonconformity = [[] for _ in range(scores.shape[1])]
        # for i in range(scores.shape[0]):
        #     # Get the starting time of the sequence
        #     start_time = self.cali_xh[i][self.args.L:][0]  # adjust this based on your data format
            
        #     for j in range(scores.shape[1]):
        #         # Compute the interval index
        #         interval = int(start_time * 4 + j) % 96  # assumes 15-minute intervals
                
        #         # Append the score to the corresponding interval
        #         self.nonconformity[interval].append(scores[i, j])
        # self.nonconformity = np.array(self.nonconformity).T
        # print(self.nonconformity.shape)


    ################# ORIGINAL PREDICT FUNCTION ######################
    # To change this back to original algorithm
    # Remove for loop to calculate scores_time
    # just leave it as scores
    # then calculate alphas using scores
    def predict(self, X_test=None, epsilon=0.1):
        pred_y = self.model(self.copula_yl, self.copula_xw, self.copula_xt)
        scores = torch.norm((pred_y-self.copula_yh), p=2, dim = -1).cpu().detach().numpy()
        alphas = []

        # scores_time = [[] for _ in range(scores.shape[1])]
        # for i in range(scores.shape[0]):
        #     # Get the starting time of the sequence
        #     start_time = self.copula_xh[i][self.args.L:][0]  # adjust this based on your data format
            
        #     for j in range(scores.shape[1]):
        #         # Compute the interval index
        #         interval = int(start_time * 4 + j) % 96  # assumes 15-minute intervals
                
        #         # Append the score to the corresponding interval
        #         scores_time[interval].append(scores[i, j])
        # scores_time = np.array(scores_time).T
        # print(scores_time.shape)
        # print(self.nonconformity.shape)

        for i in range(scores.shape[0]):
            a = (scores[i]>self.nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)
        # x_candidates = np.linspace(0.0001, 0.999, num=300)
        # x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]
        # x_sorted = sorted(list(zip(x_fun, x_candidates)))

        threshold = search_alpha(alphas, epsilon, epochs=800)

        mapping = {i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])}
        
        quantile = []
        mapping_shape = self.nonconformity.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape -1
            quantile.append(mapping[i][idx])

        radius = np.array(quantile)

        # y_pred = self.model(X_test)
        
        # self.results_dict[epsilon] = {'y_pred': y_pred, 'radius': radius}

        # return y_pred, radius
        return radius
    ##################################################################


    # def predict(self, X_test=None, epsilon=0.1):
    #     # Generate predictions and calculate nonconformity scores
    #     pred_y = self.model(self.copula_yl, self.copula_xw, self.copula_xt)
    #     scores = torch.norm((pred_y-self.copula_yh), p=2, dim = -1).detach().numpy()
    #     print(scores.shape)
        
    #     # Compare nonconformity scores to the calibration scores for the same hour
    #     # alphas = []
    #     # for i in range(scores.shape[0]):
    #     #     # Get the starting time of the sequence
    #     #     start_hour = self.copula_xh[i][self.args.L:][0]  # adjust this based on your data format
    #     #     alpha = []
    #     #     for j in range(scores.shape[1]):
    #     #         hour = math.floor((start_hour + 1 / 4) % 24)  # assumes 15-minute intervals
    #     #         a = (scores[i, j] > self.nonconformity[hour]).mean()
    #     #         alpha.append(a)
    #     #     alphas.append(alpha)
        
    #     # Compute the starting hours of each sequence
    #     start_times = self.copula_xh[:, self.args.L:]  # adjust this based on your data format
    #     intervals = ((start_times * 4) % 96).detach().numpy().astype(int)
    #     print(intervals.shape)

    # # Calculate alphas using advanced indexing
    #     print("computing alphas...")
    #     # Calculate the maximum length of the sub-lists
    #     # max_len = max(map(len, self.nonconformity))
    #     # print(max_len)
    #     # # Pad shorter lists with np.nan and convert to 2D numpy array
    #     # self.nonconformity = np.array([np.pad(interval_scores, (0, max_len - len(interval_scores)), constant_values=np.nan) 
    #     #                                 for interval_scores in self.nonconformity])
    #     print(self.nonconformity.shape)

    #     # scores_flat = scores.flatten()
    #     # intervals_flat = intervals.flatten()

    #     # Calculate alphas
    #     alphas = (scores > self.nonconformity[intervals])

    #     # Reshape alphas back to the original shape of scores
    #     # alphas = alphas_flat.reshape(scores.shape).mean(axis=0)

    #     print("get stuck here?")

    #     # alphas = np.array(alphas)


    #     # x_candidates = np.linspace(0.0001, 0.999, num=300)
    #     # x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]
    #     # x_sorted = sorted(list(zip(x_fun, x_candidates)))

    #     print("Optimising loss function...")
    #     threshold = search_alpha(alphas, epsilon, epochs=800)

    #     mapping = {i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])}
        
    #     quantile = []
    #     mapping_shape = self.nonconformity.shape[0]

    #     print("Computing radius...")
    #     for i in range(alphas.shape[1]):
    #         idx = int(threshold[i] * mapping_shape) + 1
    #         if idx >= mapping_shape:
    #             idx = mapping_shape -1
    #         quantile.append(mapping[i][idx])

    #     radius = np.array(quantile)

    #     # y_pred = self.model(X_test)
        
    #     # self.results_dict[epsilon] = {'y_pred': y_pred, 'radius': radius}

    #     # return y_pred, radius
    #     return radius



    def calc_area(self, radius):
        
        area = sum([np.pi * r**2 for r in radius])

        return area


    def calc_area_3d(self, radius):
        
        area = sum([4/3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):
        
        area = sum(radius)

        return area


    # For original function
    # let scores be testnonconformity
    # remove for loop that calculates testnonconformity
    # calculate circle_covs using scores instead of testnonconformity
    def calc_coverage(self, radius, y_pred, y_test, xh_test):

        # Need to fix dimensions as it is returning one coverage per batch
        # Needs to be one coverage per data point?

        scores = torch.norm((y_pred-y_test), p=2, dim = -1).cpu().detach().numpy()

        # testnonconformity = [[] for _ in range(scores.shape[1])]

        # for i in range(scores.shape[0]):
        #     # Get the starting time of the sequence
        #     start_time = xh_test[i][self.args.L:][0]  # adjust this based on your data format
            
        #     for j in range(scores.shape[1]):
        #         # Compute the interval index
        #         interval = int(start_time * 4 + j) % 96  # assumes 15-minute intervals
                
        #         # Append the score to the corresponding interval
        #         testnonconformity[interval].append(scores[i, j])
        # testnonconformity = np.array(testnonconformity).T

        # print("radius:", radius.shape)
        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(scores[:,j]<radius[j])


        # coverage = circle_covs.mean(axis=-1)
        circle_covs = np.array(circle_covs)
        # print("circle:", circle_covs.shape)
        # coverage = np.mean(circle_covs)

        coverage = np.mean(np.all(circle_covs, axis=0))
        # print("cov:", coverage.shape)
        return coverage


    def calc_coverage_3d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred, y_test)


    def calc_coverage_1d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred, y_test)


