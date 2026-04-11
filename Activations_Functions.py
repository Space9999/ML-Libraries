import numpy as np
import math

def linear(output):
    return output

def sigmoid(output):
    return (1 + (1 / math.exp(-output)))

def sigmoid_grad(output):
    return sigmoid(output) * (1 - sigmoid(output))

def tanh(output):
    return math.tanh(output)

def tanh_grad(output):
    return 1 - np.power(tanh(output), 2)

def relu(output):
    if (output < 0):
        return 0
    return output

def relu_grad(output):
    if (output < 0):
        return 0
    return 1

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
    return alpha * (math.exp(output) - 1)

# Alpha and gamma is a hyperparameter
def selu(output, alpha, gamma):
    if (output > 0):
        return output * gamma
    return gamma * alpha * (math.exp(output) - 1)

def soft_plus(output):
    return math.log10(1 + math.exp(output))
