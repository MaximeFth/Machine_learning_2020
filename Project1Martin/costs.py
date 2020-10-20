# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
def compute_loss(y, tx, w):

    return np.square(np.subtract(y,tx@w)).mean()


