# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    # get the number of datapoint dedicated to training
    Tr=round(ratio*x.shape[0])
    Te=round(x.shape[0]-Tr)
    
    # shuffle the two lists with same order
    temp = list(zip(x, y)) 
    np.random.shuffle(temp) 
    res1, res2 = zip(*temp)

    x_training=np.asarray([res1[:Tr]][0])
    y_training=np.asarray([res2[:Tr]][0])
    x_test=np.asarray([res1[Tr:]][0])
    y_test=np.asarray([res2[Tr:]][0])
    
    return x_training,y_training,x_test,y_test
