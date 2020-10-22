# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
def compute_loss(y, tx, w):
    """
    compute mse cost
    
    :param y: labels
    :param tx: features
    :param w: weights
    
    :return mse: mean square loss 
    """
    mse = np.square(np.subtract(y,tx@w)).mean()
    return mse


