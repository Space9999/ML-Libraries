import numpy as np

epsilon = 1e-15

def mse(y, y_pred):
    return np.sum((y - y_pred) ** 2) / np.size(y)

def mse_grad(y, y_pred):
    return np.sum(2 * (y - y_pred)) / np.size(y)

def mae(y, y_pred):
    return np.sum(abs(y - y_pred)) / np.size(y)

def mae_grad(y, y_pred):
    mae = mae(y, y_pred)
    if (mae < 0):
        return -1
    if (mae > 0):
        return 1
    return 0

def mbe(y, y_pred):
    return np.sum(y - y_pred) / np.size(y)

def mbe_grad():
    return 1

# Delta is a hyperparameter
def Huber(y, y_pred, delta):
    mae = mae(y, y_pred)
    if (mae <= delta):
        return (0.5 * mse(y, y_pred))
    return delta * mae - (0.5 * delta**2)

def Huber_grad(y, y_pred, delta):
    mae = mae(y, y_pred)
    if (mae <= delta):
        return mbe(y, y_pred)
    return delta * mae_grad(y, y_pred)

# For binary classification (y_pred is either 1 or -1)
def cross_entropy(y, y_pred):
    return -np.sum(y * np.log10(y_pred) + (1 - y) * np.log10(1 - y_pred)) / np.size(y)

# For multiclass classification (y_pred has to be the probabilities of the given class and y must be the true labels)
def categorical_cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1 - (epsilon)) # Prevents y_pred from being 0
    return -np.sum(y * np.log10(y_pred), axis = 1) / np.size(y) # Axis specifies the column in which the class and probabilities would be in

# For binary classification
def hinge(y, y_pred):
    l = 0
    for i in range(np.size(y)):
        l += max(0, 1 - y[i] * y_pred[i])
    return l / np.size(y)

# P and Q must be probability distributions
def kl_divergence(P, Q):
    P = np.clip(P, epsilon, 1) # Ensure non-zero values
    Q = np.clip(Q, epsilon, 1)    
    return np.sum(P * np.log10(P / Q))
