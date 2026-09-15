import numpy as np

def linear(output):
    return output

def sigmoid(output):
    return 1 / (1 + np.exp(-output))

def sigmoid_grad(output):
    return sigmoid(output) * (1 - sigmoid(output))

def tanh(output):
    return np.tanh(output)

def tanh_grad(output):
    return 1 - np.power(tanh(output), 2)

def relu(output):
    return np.where(output >= 0, output, 0)

def relu_grad(output):
    return np.where(output >= 0, 1, 0)

def leaky_relu(output, alpha = 0.01):
    if (output < 0):
        return alpha * output
    return output

def leaky_relu_grad(output, alpha = 0.01):
    if (output < 0):
        return alpha
    return 1

# Alpha is a hyperparameter
def elu(output, alpha):
    if (output > 0):
        return output
    return alpha * (np.exp(output) - 1)

# Alpha and gamma is a hyperparameter
def selu(output, alpha, gamma):
    if (output > 0):
        return output * gamma
    return gamma * alpha * (np.exp(output) - 1)

def soft_plus(output):
    return np.log10(1 + np.exp(output))

def softmax(output):
    max_value = np.max(output, axis = -1, keepdims = True)
    max_value[max_value == float("-inf")] = 0
    e_x = np.exp(output - max_value)
    divisor = np.sum(e_x, axis = -1, keepdims = True)
    divisor[divisor == 0] = 1e-15 # Some small value to prevent 0/0
    return e_x / divisor

def softmax_grad(grad_output, softmax_output):
    inner = np.sum(grad_output * softmax_output, axis = -1, keepdims=True)
    return softmax_output * (grad_output - inner)
