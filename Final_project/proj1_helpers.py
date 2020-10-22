"""some helper functions for project 1."""
import csv
import numpy as np

# least square gradient helpers functions
def compute_gradient(y, tx, w):
    """
    Compute the gradient.
    
    :param y: labels
    :param tx: features
    :param w: weights
    :return grad: gradient
    """
    grad = (-1/len(y))*tx.T@(y-tx@w)
    return grad

# least square stochastic gradient helpers functions
def compute_stoch_gradient(y, tx, w):
    """
    Compute the gradient.
    
    :param y: labels
    :param tx: features
    :param w: weights
    :return grad: gradient
    """
    e = y-tx@w
    grad = -1/len(y)*tx.T@e
    return grad
# logistic regression and regularized helpers functions

def sigmoid(z):
    """
    sigmoid function
    
    :param z: 
    :return z:
    """
    return 1/(1+np.exp(-z))
def update_weights(y, tx, w, gamma):
    """
    Update weights function for logistic regression
    
    :param tx: features matrix
    :param y: labels vector
    :param w: weights
    :param gamma: learning rate
    
    :return w: new updated weights
    """ 
    #probabilities array that the label is 1
    probabilities = sigmoid(np.dot(tx, w))
    gradient = np.dot(tx.T,  probabilities - y)
    w -= gradient*gamma / len(tx)
    return w

def LR_loss_function(y, tx, w):
    """
    Computes logistic loss
    
    :param tx: features matrix
    :param y: labels vector
    :param w: weights
    
    :return loss: logistic loss
    """ 
    #probabilities array that the label is 1
    probabilities = sigmoid(np.dot(tx, w))
    #the error when label=1
    error1 = -y*np.log(probabilities)
    #the error when label=-1
    error2 = (1-y)*np.log(1-probabilities)
    #return average of sum of costs
    loss = (error1-error2).mean()
    return loss



def reg_LR_update_weights(y, tx, w, gamma, lambda_):
    """
    Update weights function for  regularized logistic regression
    
    :param tx: features matrix
    :param y: labels vector
    :param w: weights
    :param gamma: learning rate
    :param lambda_: regulizer
    
    :return w: new updated weights
    """ 
    # probabilities array that the label is 1
    probabilities = sigmoid(np.dot(tx, w))
    gradient = np.dot(tx.T,  probabilities - y) + lambda_ * w
    w -= gradient*gamma / len(tx)
    return w


def reg_LR_loss_function(y, tx, w, lambda_):
    """
    Computes logistic loss
    
    :param tx: features matrix
    :param y: labels vector
    :param w: weights
    :param lambda_: regulizer
    
    :return w: logistic loss
    """ 
    # probabilities array that the label is 1
    probabilities = sigmoid(np.dot(tx, w))
    # the error when label=1
    error1 = -y*np.log(probabilities)
    # the error when label=0
    error2 = (1-y)*np.log(1-probabilities)
    # return average of sum of costs
    return (error1-error2).mean()+lambda_/2*np.dot(w.T,w)/ len(tx)





def compute_accuracy(y_pred, y):
    """
    compute the accuracy
    
    :param y_pred: predictions
    :param y: real labels
    
    :return acc: accuracy
    """
    # y_pred - y & count 0
    arr = np.array(y_pred) - np.array(y)
    acc = np.count_nonzero(arr==0) / len(y)
    return acc

def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    
    :param y: labels
    :param k_fold: number of folds
    :param seed: seed for randomization
    
    :return k_indices: indices 
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def replace(tX, value):
    """
    Replaces invalid values with the mean of all the values in the cooresponding feature 

    :param tX: features
    :param value: value to replace
    :return tX: tX with replaced values
    """
    for i in range(tX.shape[1]):
        data = tX[:, i].copy()
        np.delete(data, np.where(data == value))  
        data_median = np.median(data)  
        tX[tX[:, i] == value,i] = data_median  
    return tX

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    
    :param x: matrix 
    :param degree: degree of expansion
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly




    
def evaluate(tx, wf, degree):
    
    """
    function to evaluate weights over all the train model
    
    :param tx: train features
    :param wf: wights to evaluate
    :param degree: degree of expansion
    :return acc: accuracy of the weights over the train model
    """
    if degree is not None:
        tx =build_poly(tx, degree)
    if isinstance(wf, list):
        wk =np.mean(wf, axis =0)

    else:
        wk = wf
                
    y_pred = predict_labels(wk, tx)
    acc = compute_accuracy(y_std*2-1, y_pred)
    return acc
def remove_outliers_IQR(tx, y_, high,low):
    """
    removes outliers using IQR
    
    :param tx: features
    :param y: labels
    :param high: high IQR
    :param low: low IQR
    :returns tX, Y: features and labels without outliers
    """
    Q1 = np.quantile(tx,low,axis = 0)
    Q3 = np.quantile(tx,high, axis = 0)
    IQR = Q3 - Q1
    tX_no_outliers = tx[~((tx < (Q1 - 1.5 * IQR)) |(tx > (Q3 + 1.5 * IQR))).any(axis=1)]
    y_no_outliers = y_[~((tx < (Q1 - 1.5 * IQR)) |(tx > (Q3 + 1.5 * IQR))).any(axis=1)]
    return tX_no_outliers, y_no_outliers



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def standardize(x, mean=None, std= None):
    """
    standardization function
    
    :param x: features
    :param mean: mean to standardize with
    :param std: standard deviation to standardize with
    :return x,mean,std: standardized features, and the mean and std they have been standardized with
    """
    
    if mean is None:
        mean = np.mean(x, axis=0)
    x = x - mean
    if std is None:
        std = np.std(x, axis=0)
    x = x / std
    return x, mean, std


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
    

def cross_validation(y, x, k_indices,k, regression_method, **kwargs):
    """
    Computes cross validation on a given data set using a given regression method, and computes the
    weights, the train loss, the test loss, and the train and loss accuracy
    if the degree is not none, it will perform feature expansion on the data set
    
    :param y: labels vector
    :param tx: features matrix
    :param k_indices: k_fold already randomly computed indices
    :param degree: degree of polynomial expansion
    :param logistic: boolean; if true, the loss used is the logistic one
    :param **kwargs: differents parameters such as the regulizer lambda or the learning rate gamma
    """
    degree = kwargs["degree"]
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
    

    w_initial = np.zeros(x_train.shape[1])
    kwargs = kwargs
    kwargs['initial_w'] = w_initial

    aug = False
    if regression_method is reg_logistic_regression:
        w, loss_train = regression_method(y_train, x_train, kwargs['initial_w'],  kwargs['max_iter'], kwargs['gamma'], kwargs['lambda_'])
        loss_test = reg_LR_loss_function(y_test, x_test, w ,kwargs['lambda_'])
        
    elif regression_method is logistic_regression:
        w, loss_train = regression_method(y_train, x_train, kwargs['initial_w'],  kwargs['max_iter'], kwargs['gamma'])
        loss_test = LR_loss_function(y_test, x_test, w)
        
    elif regression_method is least_square:
        w, loss_train = regression_method(y = y_train, tx = x_train)
        loss_test = compute_loss(y_test, x_test, w)
        
    elif regression_method is ridge_regression:
        w, loss_train = regression_method(y_train,x_train, kwargs['lambda_'])
        loss_test = compute_loss(y_test, x_test, w)
    elif regression_method is least_squares_SGD: 
        aug = True
        y_test = (y_test*2)-1
        y_train = (y_train*2)-1
        w, loss_train = regression_method(y_train, x_train, kwargs['initial_w'],kwargs['batch_size'],  kwargs['max_iter'], kwargs['gamma'])
        loss_test = compute_loss(y_test, x_test, w)
    else:
        aug = True
        y_test = (y_test*2)-1
        y_train = (y_train*2)-1
        w, loss_train = regression_method(y_train, x_train, kwargs['initial_w'],  kwargs['max_iter'], kwargs['gamma'])
        loss_test = compute_loss(y_test, x_test, w)

    if not aug:
        y_test = (y_test*2)-1
        y_train = (y_train*2)-1
    y_train_pred = predict_labels(w, x_train)
    y_test_pred = predict_labels(w, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)
    return w, loss_train, loss_test, accuracy_train, accuracy_test

def train(model,y,tx,seed=0, **kwargs):
    """
    regularized logistic regression function 
    
    :param Model: model that we'll use
    :param y: labels vector
    :param tx: features matrix
    :param k_fold: number of folds
    :param degree: degree of polynomial expansion
    :param seed: random seed for cross validation split
    :param **kwargs: multiple possible parameters
    
    :return wf: final weights 
    """    
    weights = []
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []
    k_fold = kwargs["k_fold"]
    
    if kwargs["degree"] == 0:
        kwargs["degree"] = None
    degree = kwargs["degree"]
    

    logistic = False
    if model is logistic_regression or model is reg_logistic_regression:
        logistic = True
    
    k_indices = build_k_indices(y, k_fold, seed)
    for k in tqdm(range(k_fold)):
        w, loss_train, loss_test, accuracy_train, accuracy_test = cross_validation(y, tx, k_indices,k, model, **kwargs)
        weights.append(w)
        losses_train.append(loss_train)
        losses_test.append(loss_test)
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)
    leg = ["train loss "+str(i) for i in range(k_fold)]
    plt.legend(leg)
    plt.plot(losses_train[-1])    
    print("<-"+"-"*75+"->")
    if degree is not None:
        degree = int(degree)
    print("{:15.14}|{:15.14}|{:15.14}|{:15.14}|{:15.14}".format("Train losses","Test losses","Train accuracy","Test Accuracy", "Evaluation"))
    for i in range(k_fold):
        if model is least_square or model is ridge_regression:
            print("{:< 15f}|{:< 15f}|{:< 15f}|{:< 15f}|{:< 15f}".format(losses_train[i], losses_test[i] ,accuracies_train[i], accuracies_test[i], evaluate(tX_std, np.array(weights[i]), degree)))
        else:
            print("{:< 15f}|{:< 15f}|{:< 15f}|{:< 15f}|{:< 15f}".format(losses_train[i][-1], losses_test[i] ,accuracies_train[i], accuracies_test[i], evaluate(tX_std, np.array(weights[i]), degree)))
        print("{:15.1}|{:15.14}|{:15.14}|{:15.14}|{:15.14}".format("","","","",""))
    print("<-"+"-"*75+"->")
    print(f"evaluation mean w: {evaluate(tX_std, weights, degree)}")

    return weights, sum(accuracies_train)/len(accuracies_train),sum(accuracies_test)/len(accuracies_test)
    
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