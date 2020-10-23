import proj1_helpers as ph
import numpy as np
import matplotlib.pyplot as plt

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
    """
    Least square gradient descent
    
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iter: max number of iterations
    :param gamma: learning rate
    :return ws: weights
    :return ls: loss
    """    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iter):
        gradient = ph.compute_gradient(y,tx,w)
        loss = ph.compute_loss(y,tx,w)
        w = w-gamma*gradient
        ws.append(w)
        losses.append(loss)
        
    return np.array(ws)[-1], np.array(losses)
def least_squares_SGD(y, tx, initial_w, batch_size, max_iter, gamma):
    """
    Least square stochastic gradient descent
    
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param batch_size: 1 if sgd
    :param max_iter: max number of iterations
    :param gamma: learning rate
    :return ws: weights
    :return ls: loss
    """   
    ws = []
    losses = []
    w = initial_w
    for n_iter in range(max_iter):
        # compute random batch
        a = ph.batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        a = list(a)
        tx2, y2 = a[0][1], a[0][0]
        # compute gradient & loss
        grad = ph.compute_stoch_gradient(y2,tx2,w)
        loss= ph.compute_loss(y2, tx2, w)
        # update gradient
        w = w-gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)


    return np.array(ws)[-1], np.array(losses)



def least_square(y, tx):
    """
    Solves the closed form least square equation to obtain optimal weights
    
    :param y: labels
    :param tx: features
    :returns w,l: weights and loss of the model
    """
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    l = ph.compute_loss(y, tx, w)
    return w, l

def ridge_regression(y, tx, lambda_):
    """
    Solves the closed form of Ridge regression equation to obtain optimal weights
    
    :param y: labels
    :param tx: features
    :param lambda_: regulizer
    :returns w,l: weights and loss of the model
    """
    w = np.linalg.solve(tx.T@tx+lambda_*np.eye(tx.shape[1]),tx.T@y)
    l = ph.compute_loss(y, tx, w)
    return w, l


def logistic_regression(y,tx, initial_w,  max_iter, gamma):
    """
    Logistic regression function
    
    :param tx: features matrix
    :param y: labels vector
    :param initial_w: initial weights
    :param max_iter: number of iterations
    :param gamma: learning rate

    :return ls: last loss  computed
    :return ws: last weights computed
    """ 
    losses = []
    ws = []
    for iter_n in range(max_iter):
        w = ph.update_weights(y, tx, initial_w, gamma)
        loss = ph.LR_loss_function(y, tx, w)
        losses.append(loss)
        ws.append(w)
    ls, wes  = np.array(losses), np.array(ws)
    return wes[-1],ls


def reg_logistic_regression(y,tx, initial_w,max_iter, gamma,lambda_):
    """
    Regularized logistic regression function
    
    :param tx: features matrix
    :param y: labels vector
    :param initial_w: initial weights
    :param max_iter: number of iterations
    :param gamma: learning rate
    :param lambda_: regulizer

    :return ls: last loss  computed
    :return ws: last weights computed
    """ 
    losses = []
    ws = []
    for iter_n in range(max_iter):
        if(iter_n > 800):
            gamma = gamma-gamma/30
        w = ph.reg_LR_update_weights(y, tx, initial_w, gamma,lambda_)
        loss = ph.reg_LR_loss_function(y, tx, w, lambda_)
        losses.append(loss)
        ws.append(w)
    ls, wes  = np.array(losses), np.array(ws)
    return wes[-1],ls

















