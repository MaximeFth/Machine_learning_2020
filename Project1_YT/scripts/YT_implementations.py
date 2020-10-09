# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *

# ------------------------------------------------------------- #
def calculate_mse(e):
    """Calculate the mse for vector e"""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.
    Input:
    - y, tx: two iterables
    - shuffle: data can be randomly shuffled
    Output:
    - an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
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

def sigmoid(t):
    """apply sigmoid function on t."""
    #1 / (1 +np.exp(-t))
    return np.exp(-np.logaddexp(0, -t))


def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    #eps = 1e-12
    eps = np.finfo(float).eps
    sigma = sigmoid(tx.dot(w))
    loss = -(y.T.dot(np.log(sigma+eps)) + (1-y).T.dot(np.log(1-sigma+eps))).squeeze()
    return loss


def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma = sigmoid(tx.dot(w))
    gradient = tx.T.dot(sigma - y)
    return gradient

# ------------------------------------------------------------- #

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
    
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    lambda_prime = lambda_ * 2 * tx.shape[0]
    lambda_I = lambda_prime * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_I
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        loss = compute_logistic_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # compute regularized loss, gradient
        loss = compute_logistic_loss(y, tx, w) + lambda_*w.T.dot(w).squeeze()
        grad = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
        # gradient w by descent update
        w = w - gamma * grad

    return w, loss

# ------------------------------------------------------------- #

def standardization(x, mean, std):
    return (x - mean) / (std + np.finfo(float).eps)


def compute_accuracy(y_pred, y):
    # y_pred - y & count 0
    arr = np.array(y_pred) - np.array(y)
    return np.count_nonzero(arr==0) / len(y)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def cross_validation(y, x, k_indices, k, degree, logistic, regression_method, **kwargs):
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    
    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]
    
    if degree != None:
        x_train = build_poly(x_train, degree)
        x_test = build_poly(x_test, degree)

    if logistic == True:
        w_initial = np.zeros(x_train.shape[1])
        kwargs = kwargs
        kwargs['initial_w'] = w_initial

    w, loss_train = regression_method(y = y_train, tx = x_train, **kwargs)

    loss_test = compute_loss(y_test, x_test, w)
    
    y_train_pred = predict_labels(w, x_train)
    y_test_pred = predict_labels(w, x_test)
    
    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)
    
    return w, loss_train, loss_test, accuracy_train, accuracy_test


