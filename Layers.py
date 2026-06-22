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
    'soft_plus': Activations_Functions.soft_plus,
    'softmax': Activations_Functions.softmax
}

activation_gradients = {
    'tanh_gradient': Activations_Functions.tanh_grad,
    'sigmoid_gradient': Activations_Functions.sigmoid_grad,
    'relu_gradient': Activations_Functions.relu_grad,
    'leaky_relu_gradient': Activations_Functions.leaky_relu_grad,
    'softmax_gradient': Activations_Functions.softmax_grad
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
            self.weight_bias = self.weight_bias_optimizer.update(self.weight_bias, grad_weight_bias)
        
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

class Activation(Layer):

    def __init__(self, name):
        self.activation_function = activation_functions[name]
        self.activation_name = name
        self.trainable = True   
    
    def get_layer_name(self):
        return "(%s) Activation" % (self.activation_name)
    
    def forward_pass(self, X, trainable = True):
        self.layer_input = X
        return self.activation_function(X)
    
    def backward_pass(self, accumulated_grad):
        if self.activation_name == 'softmax': # Edge case for softmax function to use full jacobian matrix
            softmax_output = self.activation_function(self.layer_input)
            return activation_gradients['softmax_gradient'](accumulated_grad, softmax_output)
        return accumulated_grad * activation_gradients[self.activation_name + '_gradient'](self.layer_input)
    
    def get_output_shape(self):
        return self.input_shape

# 2D Convolution Layer
class Conv2D(Layer):

    def __init__(self, name):
        self.activation_function = activation_functions[name]
        self.activation_name = name
        self.Trainable = True
    
    def get_layer_name(self):
        return "(%s) Activation" % (self.activation_name)
    
    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight_bias.shape)
    
# Helper methods for convolution
# Equations are referenced from CS231n Stanford (https://cs231n.github.io/convolutional-networks/)
def determine_padding(filter_shape, padding_type = "same"):

    if padding_type == "no_padding":
        return (0, 0), (0, 0)
    
    # Pad such that output shape is equal to the input shape
    # Determined by the following equation
    # output_height = ((input_height + pad_h - filter_height) / stride) + 1 where output_height = input_height and stride = 1
    # Note: The above equation works for width as well
    elif padding_type == "same":
        filter_height, filter_width = filter_shape

        # For ideal results, height1 = height2 and vice versa for width
        pad_height1 = int(math.floor((filter_height - 1) / 2))
        pad_height2 = int(math.ceil((filter_height - 1) / 2))
        pad_width1 = int(math.floor((filter_width - 1) / 2))
        pad_width2 = int(math.ceil((filter_width - 1) / 2))

    return (pad_height1, pad_height2), (pad_width1, pad_width2)

def get_im2col_indices(images_shape, filter_shape, padding, stride = 1):
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_height, pad_width = padding

    # Uses equation referenced in im2col indices
    output_height = int((height + np.sum(pad_height) - filter_height) / stride + 1)
    output_width = int((width + np.sum(pad_width) - filter_width) / stride + 1)

    # Generates indices for corresponding components
    # i -> row indices
    # j -> column indices
    # k -> channel indices
    # Allows for quick lookup of elements of a certain convolution patch
    i1 = np.repeat(np.arange(filter_height), filter_width)
    i1 = np.tile(i1, channels)
    i2 = stride * np.repeat(np.arange(output_height), output_width)
    j1 = np.tile(np.arange(filter_width), filter_height * channels)
    j2 = stride * np.tile(np.arange(output_width), output_height)
    i = i1.reshape(-1, 1) + i2.reshape(-1, 1)
    j = j1.reshape(-1, 1) + j2.reshape(-1, 1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)

# Used in forward pass
# Converts image data into column shape
def image_to_column(images, filter_shape, stride, padding_type = "same"):
    filter_height, filter_width = filter_shape

    pad_height, pad_width = determine_padding(filter_shape, padding_type)

    images_padded = np.pad(images, ((0, 0), (0, 0), pad_height, pad_width), mode = "constant")
        


        


        





        
        
        



    
    