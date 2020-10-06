# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss_MSE(y, tx, w):
    N=tx.shape[0]
    err=y-tx@w
    loss=1/(2*N)*np.transpose(err)@err
    return loss

def compute_loss_MAE(y, tx, w):
    N=tx.shape[0]
    err=tx@w-y
    loss=1/N*np.sum(np.abs(err))
    return loss