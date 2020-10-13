# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_=lambda_*(2*len(tx))
    w=np.dot(np.linalg.inv(np.dot(tx.T,tx)+lambda_*np.identity(tx.shape[1])),np.dot(tx.T,y))
    err=tx@w-y
    loss=1/(2*len(tx))*np.sum(np.square(err))
    return w, loss
    raise NotImplementedError
