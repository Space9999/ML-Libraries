import numpy as np

# Simplified version of SGD with momentum for the purposes of implementation in layers
class Simplified_SGD():
    def __init__(self, learning_rate = 0.01, momentum = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_update = None

    def update(self, weight, gradient_weight):
        if self.weight_update is None:
            self.weight_update = np.zeros_like(weight)

        self.weight_update = self.momentum + self.weight_update + (1 - self.momentum) * gradient_weight

        return weight - self.learning_rate * self.weight_update

class Adam():
    def __init__(self, learning_rate = 0.001, b1 = 0.9, b2 = 0.999):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.t = 0
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2
    
    def update(self, weight, grad_weight):
        if self.m is None:
            self.m = np.zeros_like(grad_weight)
            self.v = np.zeros_like(grad_weight)
        
        self.m = self.b1 * self.m + (1 - self.b1) * grad_weight
        self.v = self.b2 * self.v + (1 - self.b2) * grad_weight ** 2

        self.t += 1
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)

        self.weight_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weight - self.weight_update

# Gradient clipping by norm
def gradient_clip(gradient, max_norm = 5.0):
    # Calculates frobenius norm for matrix: https://mathworld.wolfram.com/FrobeniusNorm.html
    gradient_norm = np.linalg.norm(gradient)
    
    if gradient_norm > max_norm:
        return gradient * (max_norm / gradient_norm)

    return gradient









    