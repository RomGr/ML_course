# -*- coding: utf-8 -*-

import numpy as np

#---------------------- Least square GD functions ------------------------

def calculate_mse(e):
    """Calculate mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss using mse or mae"""
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N=tx.shape[0]
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / N
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        grad=compute_gradient(y, tx, w)
        loss=compute_loss(y, tx, w)
        w=w-gamma*grad

    return w, loss

#---------------------- Least Squares SGD functions --------------------------

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    N=tx.shape[0]
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / N
    return grad

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
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            grad=compute_stoch_gradient(y_batch, tx_batch, w)
            loss=compute_loss(y_batch, tx_batch, w)
            w=w-gamma*grad

    return w, loss

#---------------------------- Least Squares functions ------------------------------



def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return w,loss

# --------------------------- Ridge regression -------------------------------------

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_=lambda_*(2*len(tx))
    w=np.dot(np.linalg.inv(np.dot(tx.T,tx)+lambda_*np.identity(tx.shape[1])),np.dot(tx.T,y))
    err=tx@w-y
    mse=1/(2*len(tx))*np.sum(np.square(err))
    return w, mse

# -------------------------- Logistic regression -----------------------------------

def sigmoid(t):
    """apply the sigmoid function on t."""
    return(np.exp(t)/(np.exp(t)+1))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss=0
    for i in range(tx.shape[0]):
        loss=loss+np.log(1+np.exp(tx[i].T@w))-y[i]*tx[i].T@w
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss=calculate_loss(y, tx, w)
    gradient=calculate_gradient(y, tx, w)
    w=w-gamma*gradient
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        #Do the logistic regression using Gradient Descent
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss



# -------------------------- Penalized logistic regression -----------------------------------

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient."""
    loss=calculate_loss(y, tx, w)
    loss=loss + lambda_ * np.squeeze(w.T.dot(w))

    gradient=calculate_gradient(y, tx, w)
    gradient=gradient+2* lambda_ *w

    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)

    w=w-gamma*gradient
    return loss, w


def logistic_regression_penalized_gradient_descent(y, tx, w, lambda_, max_iter, gamma, batch_size=0):
    threshold = 1e-8
    losses = []
    if batch_size != 0:
        batch=batch_iter(y,tx,batch_size)
        data=next(batch)
        y=data[0]
        tx=data[1]

    # start the logistic regression
    for iter in range(max_iter):

        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # log info
        if iter % 1000 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            print(np.linalg.norm(w))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    l=calculate_loss(y, tx, w)
    # visualization
    return w, l