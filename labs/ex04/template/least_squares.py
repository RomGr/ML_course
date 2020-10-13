# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    res=np.linalg.lstsq(tx,y,rcond=-1)
    w=res[0]
    MSE=res[1]/(2*len(tx))
    return w, MSE