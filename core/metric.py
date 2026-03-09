# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:46:22 2024

@author: LYT
"""

import numpy as np
import torch

#%% Metric (指标：皮尔逊和决定系数)
def metric(y,x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    #corrlation(皮尔逊系数)
    x_mean = np.mean(x, axis=-1, keepdims=True)
    # print("x_mean",x_mean.shape)
    y_mean = np.mean(y, axis=-1, keepdims=True)
    x_std = np.std(x, axis=-1, keepdims=True)
    y_std = np.std(y, axis=-1, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=-1,keepdims=True)/(x_std*y_std)
    # print("corr",corr.shape)

    #coefficeint of determination (r2)(决定系数)
    S_tot = np.sum((x-x_mean)**2, axis=-1, keepdims=True)
    S_res = np.sum((x - y)**2, axis=-1, keepdims=True)

    r2 = (1-S_res/S_tot)

    return torch.tensor(corr), torch.tensor(r2)

def display_results(loss, property_corr, property_r2, args, header):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()

    loss = torch.mean(torch.tensor(loss))
    
    # corr_text = " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_corr[i].squeeze()) for i in range(len(args.incident_angles))])
    # r2_text =   " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_r2[i].squeeze()) for i in range(len(args.incident_angles))])
    
    corr_text = " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_corr[i].squeeze().item()) for i in range(len(args.incident_angles))])
    r2_text =   " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_r2[i].squeeze().item()) for i in range(len(args.incident_angles))])
    
    # corr_text = " | ".join([u"0\xb0: {:.4f}".format(property_corr)])
    # r2_text =   " | ".join([u"0\xb0: {:.4f}".format(property_r2)])
    
    print("loss: {:.4f}\nCorrelation: {:s}\nr2 Coeff.  : {:s}".format(loss,corr_text,r2_text))
    
    
    
    