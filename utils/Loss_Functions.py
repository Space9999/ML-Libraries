import numpy as np
import utils.Activations_Functions as Activations_Functions

def mse(y, y_pred):
    return np.sum((y_pred - y) ** 2) / np.size(y)

def mse_grad(y, y_pred):
    return np.sum(2 * (y_pred - y)) / np.size(y)

def mae(y, y_pred):
    return np.sum(abs(y_pred - y)) / np.size(y)

def mae_grad(y, y_pred):
    mae = mae(y, y_pred)
    if (mae < 0):
        return -1
    if (mae > 0):
        return 1
    return 0

def mbe(y, y_pred):
    return np.sum(y_pred - y) / np.size(y)

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

# For binary classification
def cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y * np.log10(y_pred) + (1 - y) * np.log10(1 - y_pred), axis = 1))

def cross_entropy_grad(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y / y_pred) - (1 - y) / (1 - y_pred) / y.shape[0]

# For multiclass classification (y_pred has to be the probabilities of the given class and y must be the true labels)
def categorical_cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.ma.mean(np.ma.sum(y * np.ma.log(y_pred), axis = 1))

def categorical_cross_entropy_grad(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y) / y.shape[0]

# Helper method for sparse cross entropy functions
def get_probabilities_and_indices(y, y_pred):
    probs = Activations_Functions.softmax(y_pred)
    batch_size, sequence_len = y.shape

    batch_indices = np.arange(batch_size)[:, np.newaxis]
    sequence_indices = np.arange(sequence_len)[np.newaxis, :]

    return batch_indices, sequence_indices, probs

def sparse_categorical_cross_entropy(y, y_pred, mask):
    batch_indices, sequence_indices, probs = get_probabilities_and_indices(y, y_pred)

    target_probs = probs[
        batch_indices,
        sequence_indices, y
    ]
    return -np.sum(np.log(np.clip(target_probs[mask], 1e-15, 1.0))) / np.sum(mask)

def sparse_categorical_cross_entropy_grad(y, y_pred, mask):
    batch_indices, sequence_indices, probs = get_probabilities_and_indices(y, y_pred)

    gradient = probs.copy()

    gradient[
        batch_indices,
        sequence_indices, y
    ] -= 1.0
    gradient = (gradient * mask) / np.sum(mask)

    return gradient
    
# For binary classification
def hinge(y, y_pred):
    l = 0
    for i in range(np.size(y)):
        l += max(0, 1 - y[i] * y_pred[i])
    return l / np.size(y)

# P and Q must be probability distributions
def kl_divergence(P, Q):
    P = np.clip(P, 1e-15, 1) # Ensure non-zero values
    Q = np.clip(Q, 1e-15, 1)    
    return np.sum(P * np.log10(P / Q))
