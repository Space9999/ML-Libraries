import numpy as np
import math
import copy
import Activations_Functions

activation_functions = {
    'tanh': Activations_Functions.tanh,
    'linear': Activations_Functions.linear,
    'sigmoid': Activations_Functions.sigmoid,
    'relu': Activations_Functions.relu,
    'leaky_relu': Activations_Functions.leaky_relu,
    'elu': Activations_Functions.elu,
    'selu': Activations_Functions.selu,
    'soft_plus': Activations_Functions.soft_plus
}

activation_gradients = {
    'tanh_gradient': Activations_Functions.tanh_grad,
    'sigmoid_gradient': Activations_Functions.sigmoid_grad,
    'relu_gradient': Activations_Functions.relu_grad,
    'leaky_relu_gradient': Activations_Functions.leaky_relu_grad,
}

class Layer(object):

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_layer_name(self):
        return self.__class__.__name__ # Returns name of the class
    
    def parameters(self):
        return 0
    
    def forward_pass(self, input, is_training):
        raise NotImplementedError()
    
    def backward_pass(self, input, is_training):
        raise NotImplementedError()
    
    def get_output_shape(self):
        raise NotImplementedError()

# A fully connected neural network layer
# Reference for all initial weight calculatons for all layers: https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
class Dense(Layer):

    def __init__(self, n_units, input_shape = None):
        self.input_shape = input_shape # Note: input_shape is a single digit specifying the number of items in a input for dense layers 
        self.n_units = n_units
        self.layer_input = None
        self.trainable = True
        self.weight = None
        self.weight_bias = None
    
    def initialize_layer(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])

        self.weight = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))

        self.weight_bias = np.zeros((1, self.n_units))

        # We want seperate optimizers as bias is not necessarily proportional to weight
        self.weight_optimizer = copy.copy(optimizer)
        self.weight_bias_optimizer = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight_bias.shape)
    
    def forward_pass(self, input, training = True):
        self.layer_input = input
        return input.dot(self.weight) + self.weight_bias
    
    def backward_pass(self, accumulated_gradient):
        # Save weight from forward pass
        weight = self.weight

        if self.trainable:
            # Calculation for applying gradient to weight and bias
            """The weight bias is calculated through the sum of gradient vector components 
            as each portion of weight gradient will contribute to a change in bias"""
            grad_weight = self.layer_input.T.dot(accumulated_gradient)
            grad_weight_bias = np.sum(accumulated_gradient, axis = 0, keepdims = True)

            self.weight = self.weight_optimizer.update(self.weight, grad_weight)
            self.weight_bias = self.weight_bias_optimizer.update(self.weight, grad_weight_bias)
        
        accumulated_gradient = accumulated_gradient.dot(weight.T)
        return accumulated_gradient

    def get_output_shape(self):
        return (self.n_units, )

# Recurrent neural network layer
class RNN(Layer):

    """Bp_time_steps stands for backpropagation time steps meaning the amount of time steps
    the gradient will be propagated for depending on the gradient"""
    def __init__(self, n_units, activation_gradient = 'tanh_gradient', bp_time_steps = 5, input_shape = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation_gradient = activation_gradients[activation_gradient]()
        self.trainable = True
        self.bp_time_steps = bp_time_steps
        self.weight_previous = None # Previous state is related to hidden state
        self.weight_input = None 
        self.weight_output = None 
    
    def initialize_layer(self, optimizer):
        input_dim = self.input_shape[1] # Input dimension is the number of columns

        limit = 1 / math.sqrt(input_dim)
        self.weight_input = np.random.uniform(-limit, limit, (self.n_units, input_dim)) # Input to hidden weights
        limit = 1 / math.sqrt(self.n_units)
        self.weight_output = np.random.uniform(-limit, limit, (self.n_units, input_dim)) # Hidden to output weights
        self.weight_previous = np.random.uniform(-limit, limit, (self.n_units, self.n_units)) # Hidden to hidden weights (recurrent weights)

        self.weight_previous_optimizer = copy.copy(optimizer)
        self.weight_input_optimizer = copy.copy(optimizer)
        self.weight_output_optimizer = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.weight_previous.shape) + np.prod(self.weight_input.shape) + np.prod(self.weight_output.shape)
    
    def forward_pass(self, input, training = True):
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape

        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps + 1, input_dim))

        # Set last time step (t = -1) to zeros in order to prevent loop from crashing at t = 0
        self.states[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = input[:, t].dot(self.weight_input.T) + self.states[:, t - 1].dot(self.weight_previous.T)
            self.states = self.activation(self.state_input[:, t])
            self.outputs = self.states[:, t].dot(self.weight_output.T)
        
        return self.outputs
    
    def backward_pass(self, accumulated_grad):
        _, timesteps, _ = accumulated_grad.shape

        weight_input_grad = np.zeros_like(self.weight_input)
        weight_output_grad = np.zeros_like(self.weight_output)
        weight_previous_grad = np.zeros_like(self.weight_previous)

        accumulated_grad_next = np.zeros_like(accumulated_grad)

        for t in reversed(range(timesteps)):
            weight_output_grad += accumulated_grad[:, t].T.dot(self.states[:, t])
            weight_state_grad = accumulated_grad[:, t].T.dot(self.weight_output) * self.activation_gradient(self.state_input[:, t])
            accumulated_grad_next[:, t] = weight_state_grad.dot(self.weight_input)

            # This is the loop where the back traversal happens
            for t2 in reversed(np.arange(max(0, t - self.bp_time_steps), t + 1)):
                weight_input_grad += weight_state_grad.T.dot(self.layer_input[:, t2])
                weight_previous_grad += weight_state_grad.T.dot(self.layer_input[:, t2 - 1])

                # Calculate gradient for previous state
                weight_state_grad = weight_state_grad.dot(self.weight_input) * self.activation_gradient(self.state_input[:, t2 - 1])

        # Update all weights for next pass
        self.weight_input = self.weight_input_optimizer.update(self.weight_input, weight_input_grad)
        self.weight_output = self.weight_output_optimizer.update(self.weight_output, weight_output_grad)
        self.weight_previous = self.weight_previous_optimizer.update(self.weight_previous, weight_previous_grad)

        return accumulated_grad_next
    
    def get_output_shape(self):
        return self.input_shape
        



        


        





        
        
        



    
    