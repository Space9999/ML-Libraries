import numpy as np
import math
import random
"""
class Gradient_Descent:

    # In this case, the gradient function can only be for a single variable function (as the value must be a number)
    def gradient_descent(start, learning_rate, gradient, iterations, tolerance = 1e-6):
        input = start
        history = [input]

        for i in range(iterations):
            gradient_val = gradient(input)
            new_input = input - learning_rate * gradient_val

            if (abs(new_input - input) < tolerance):
                print(f"Converged at iteration {i + 1}")
                break

            input = new_input
            history.append(input)
        
        return input, history

    np.random.seed(42) # Generate random array of X and y variables as a example
    X = 2 * np.random.rand(200, 1)
    y = 3 + 2.5 * X + np.random.randn(200, 1)

    # Note: There are other flavors of gradient descent where batch_size > 1
    # In this case, the gradient function is defined as the gradient of the X and y arrays with the target vector
    def stochastic_gradient_descent(X, y, learning_rate, gradient, epochs, batch_size = 1):
        input_size = len(X)
        vector_param = np.random.randn(2, 1) # Returns a random vector in component form

        X_bias = np.c_[np.ones((input_size, 1)), X] # Attaches a default weight of one to each variable

        cost_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(input_size) # Returns a random array of size input_size
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            for i in range(0, input_size, batch_size):
                X_batch = X_shuffled[i : i + batch_size] # Creates random batches of X, y data
                y_batch = y_shuffled[i : i + batch_size]

                gradients = gradient(X_batch, y_batch, vector_param)

                diff = learning_rate * gradients
                vector_param -= diff # Note: Gradients are vectors for multiple variables
            
            predictions = np.dot(X_bias, vector_param)
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
        
        return vector_param, cost_history
    
    # Same method with tolerance
    def stochastic_gradient_descent(X, y, learning_rate, gradient, tolerance, epochs, batch_size = 1):
        input_size = len(X)
        vector_param = np.random.randn(2, 1) # Returns a random vector in component form

        X_bias = np.c_[np.ones((input_size, 1)), X] # Attaches a default weight of one to each variable

        cost_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(input_size) # Returns a random array of size input_size
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            for i in range(0, input_size, batch_size):
                X_batch = X_shuffled[i : i + batch_size] # Creates random batches of X, y data
                y_batch = y_shuffled[i : i + batch_size]

                gradients = gradient(X_batch, y_batch, vector_param)

                diff = learning_rate * gradients
                if (math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2) < tolerance):
                    break
                vector_param -= diff # Note: Gradients are vectors for multiple variables
            
            predictions = np.dot(X_bias, vector_param)
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
        
        return vector_param, cost_history

# Estimates minimum based on set bounds
class Simulated_Annealing:

    def randomize(x, step_size):
        neighbor = x[:] # Creates shallow copy of array
        index = random.randint(0, len(x) - 1)
        neighbor[index] += random.uniform(-step_size, step_size)
        return neighbor
    
    # Common function for optimization problems as it has lots of minima (2-D Rastrigin function)
    def objective_function(x):
        return 10 * len(x) + sum([(xi**2 - 10 * math.cos(2 * math.pi * xi)) for xi in x])
    
    
    def simulated_annealing(objective_function, bounds, n_iterations, step_size, temp):
        best = [random.uniform(bound[0], bound[1]) for bound in bounds]
        best_val = objective_function(best)
        current, current_val = best, best_val
        history = [best_val]

        for i in range(n_iterations):
            temp /= float(i + 1)
            candidate = Simulated_Annealing.randomize(current, step_size)
            candidate_val = objective_function(candidate)
            
            # As temp decreases, the likelihood of a change in candidate value decreases
            if candidate_val < best_val or random.uniform(0, 1) < math.exp((current_val - candidate_val) / temp):
                current, current_val = candidate, candidate_val
                if best_val > candidate_val: # Check if candidate is at a minimum compared to best val
                    best, best_val = candidate, candidate_val
                    history.append(best_val)
        
        return best, best_val, history
""" # Legacy Code (for reference purposes only)

# Simplified version of SGD with momentum for the purposes of implementation in layers
# Note: We are optimizing for min weight because the loss function must have a positive direct relationship with weight 
# Loss = error(w, other variables)
class Simplified_SGD:
    def __init__(self, learning_rate = 0.01, momentum = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_update = None

    def update(self, weight, gradient_weight):
        if self.weight_update is None:
            self.weight_update = np.zeros(np.shape(weight))

        self.weight_update = self.momentum + self.weight_update + (1 - self.momentum) * gradient_weight

        return weight - self.learning_rate * self.weight_update

# Adam optimizer
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
            self.m = np.zeros(np.shape(grad_weight))
            self.v = np.zeros(np.shape(grad_weight))
        
        self.m = self.b1 * self.m + (1 - self.b1) * grad_weight
        self.v = self.b2 * self.v + (1 - self.b2) * grad_weight ** 2

        self.t += 1
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)

        self.weight_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weight - self.weight_update









    