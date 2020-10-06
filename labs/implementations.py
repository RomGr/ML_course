# -*- coding: utf-8 -*-

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the MSE loss."""
    N=tx.shape[0]
    err=y-tx@w
    loss=1/(2*N)*np.transpose(err)@err
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N=tx.shape[0]
    err=y-tx@w
    grad=-1/N*np.transpose(tx)@err
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y, tx, w)
        loss=compute_loss(y, tx, w)
        w=w-gamma*grad
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch=batch_iter(y,tx,batch_size)
    data=next(batch)
    y=data[0]
    tx=data[1]
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y, tx, w)
        loss=compute_loss(y, tx, w)
        w=w-gamma*grad
    return w, loss



def least_squares(y, tx):
    """calculate the least squares solution."""
    res=np.linalg.lstsq(tx,y,rcond=-1)
    w=res[0]
    loss=res[1]/(2*len(tx))
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_=lambda_*(2*len(tx))
    w=np.dot(np.linalg.inv(np.dot(tx.T,tx)+lambda_*np.identity(tx.shape[1])),np.dot(tx.T,y))
    err=tx@w-y
    loss=1/(2*len(tx))*np.sum(np.square(err))
    return w, loss

