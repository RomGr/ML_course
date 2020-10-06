# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N=tx.shape[0]
    err=y-tx@w
    grad=-1/N*np.transpose(tx)@err
    return grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    # Define parameters to store w and loss
    batch=batch_iter(y,tx,batch_size)
    data=next(batch)
    y=data[0]
    tx=data[1]
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y, tx, w)
        loss=compute_loss(y, tx, w)
        w=w-gamma*grad
        ws.append(w)
        losses.append(loss)
        print("Stochastic gradient sescent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws