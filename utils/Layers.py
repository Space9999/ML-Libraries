import numpy as np
import math
import copy
import utils.Activations_Functions as Activations_Functions
from utils.Optimizers import gradient_clip

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

    def get_layer_name(self):
        return self.__class__.__name__
    
    def parameters(self):
        return 0
    
    def forward_pass(self, input, is_training):
        raise NotImplementedError()
    
    def backward_pass(self, input, is_training):
        raise NotImplementedError()
    
    def get_output_shape(self):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        self.input_shape = shape

# Reference for all initial weight calculatons for all layers: https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
# A fully connected neural network layer (equivalent with linear layer since there is no activation)
class Dense(Layer):

    def __init__(self, n_units, input_size = None, have_bias = True):
        self.input_shape = (input_size, )
        self.n_units = n_units # Number of neurons
        self.layer_input = None
        self.trainable = True
        self.weight = None
        self.weight_bias = None
        self.have_bias = have_bias
    
    def initialize_layer(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])

        self.weight = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))

        # We want seperate optimizers as bias is not necessarily proportional to weight
        self.weight_optimizer = copy.copy(optimizer)

        if self.have_bias:
            self.weight_bias_optimizer = copy.copy(optimizer)
            self.weight_bias = np.zeros((1, self.n_units))
    
    def parameters(self):
        if self.have_bias:
            return np.prod(self.weight.shape) + np.prod(self.weight_bias.shape)
        return np.prod(self.weight.shape)
    
    def forward_pass(self, input, training = True):
        self.layer_input = input
        if self.have_bias:
            return input @ self.weight + self.weight_bias
        return input @ self.weight
    
    def backward_pass(self, accumulated_gradient):
        # Save weight from forward pass
        prev_weight = self.weight

        if self.trainable:
            input_2d = self.layer_input.reshape(-1, self.layer_input.shape[-1])
            gradient_2d = accumulated_gradient.reshape(-1, accumulated_gradient.shape[-1])
            grad_weight = input_2d.T @ gradient_2d
            self.weight = self.weight_optimizer.update(self.weight, grad_weight)
            
            """The weight bias is calculated through the sum of gradient vector components 
                as each portion of weight gradient will contribute to a change in bias"""
            if self.have_bias:
                axes = tuple(range(accumulated_gradient.ndim - 1))
                grad_weight_bias = np.sum(accumulated_gradient, axis = axes, keepdims = False).reshape(1, self.n_units)
                self.weight_bias = self.weight_bias_optimizer.update(self.weight_bias, grad_weight_bias)
        
        accumulated_gradient = accumulated_gradient @ prev_weight.T
        return accumulated_gradient

    def get_output_shape(self):
        if (self.layer_input is None):
            return (self.n_units, )
        return self.layer_input.shape[:-1] + (self.n_units, )

# Learned embedding lookup table
class Embedding(Layer):

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.layer_input = None
        self.trainable = True
        self.weight = None
        self.padding_index = None
    
    def initialize_layer(self, optimizer):
        limit = 1 / np.sqrt(self.embedding_dim)

        self.weight = np.random.uniform(-limit, limit, (self.vocab_size, self.embedding_dim))
        self.weight_optimizer = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.weight.shape)
    
    def forward_pass(self, input, training = True):
        self.layer_input = input
        return self.weight[input]
    
    def backward_pass(self, accumulated_gradient):
        if self.trainable:
            grad_weight = np.zeros_like(self.weight)
            np.add.at(
                grad_weight,
                self.layer_input,
                accumulated_gradient
            )
            if (self.padding_index is not None):
                grad_weight[self.padding_index] = 0
            self.weight = self.weight_optimizer.update(self.weight, grad_weight)

        # No meaningful gradient with respect to tokens
        return None
    
    def get_weight(self): 
        return self.weight

    def set_padding_index(self, padding_index):
        self.padding_index = padding_index

# Recurrent neural network layer with truncated backpropagation
class RNN(Layer):

    """Bp_time_steps stands for backpropagation time steps"""
    def __init__(self, n_units, activation = 'tanh', activation_gradient = 'tanh_gradient', bp_time_steps = 5, input_shape = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]
        self.activation_gradient = activation_gradients[activation_gradient]
        self.trainable = True
        self.bp_time_steps = bp_time_steps
        self.weight_previous = None # Previous state is related to hidden state
        self.weight_input = None 
        self.weight_output = None 
    
    def initialize_layer(self, optimizer):
        input_dim = self.input_shape[1]

        limit = 1 / math.sqrt(input_dim)
        self.weight_input = np.random.uniform(-limit, limit, (self.n_units, input_dim)) # Input to hidden weights
        limit = 1 / math.sqrt(self.n_units) # Change in column dimension changes initialization
        self.weight_output = np.random.uniform(-limit, limit, (input_dim, self.n_units)) # Hidden to output weights
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
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step (t = -1) to zeros in order to prevent loop from crashing at t = 0
        self.states[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = input[:, t].dot(self.weight_input.T) + self.states[:, t - 1].dot(self.weight_previous.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.weight_output.T)
        
        return self.outputs
    
    def backward_pass(self, accumulated_grad):
        _, timesteps, _ = accumulated_grad.shape

        weight_input_grad = np.zeros_like(self.weight_input)
        weight_output_grad = np.zeros_like(self.weight_output)
        weight_previous_grad = np.zeros_like(self.weight_previous)

        accumulated_grad_next = np.zeros_like(accumulated_grad)

        for t in reversed(range(timesteps)):
            weight_output_grad += accumulated_grad[:, t].T.dot(self.states[:, t])
            weight_state_grad = accumulated_grad[:, t].dot(self.weight_output) * self.activation_gradient(self.state_input[:, t])
            accumulated_grad_next[:, t] = weight_state_grad.dot(self.weight_input)

            # This is the loop where the back traversal happens
            for t2 in reversed(np.arange(max(0, t - self.bp_time_steps), t + 1)):
                weight_input_grad += weight_state_grad.T.dot(self.layer_input[:, t2])
                weight_previous_grad += weight_state_grad.T.dot(self.states[:, t2 - 1])

                # Calculate gradient for previous state
                weight_state_grad = weight_state_grad.dot(self.weight_previous) * self.activation_gradient(self.state_input[:, t2 - 1])

        # Limit gradient norms (alleviates exploding gradients)
        weight_input_grad = gradient_clip(weight_input_grad)
        weight_output_grad = gradient_clip(weight_output_grad)
        weight_previous_grad = gradient_clip(weight_previous_grad)

        # Update all weights for next pass
        self.weight_input = self.weight_input_optimizer.update(self.weight_input, weight_input_grad)
        self.weight_output = self.weight_output_optimizer.update(self.weight_output, weight_output_grad)
        self.weight_previous = self.weight_previous_optimizer.update(self.weight_previous, weight_previous_grad)

        return accumulated_grad_next
    
    def get_output_shape(self):
        return self.input_shape

# Long short term memory (LSTM) which is essentially a RNN with more parameters
class LSTM(Layer):
    def __init__(self, mem_cells, input_shape, bp_time_steps = 5, activations = ['tanh', 'sigmoid'], gradients = ['tanh_gradient', 'sigmoid_gradient']):
        self.activation1 = activation_functions[activations[0]]
        self.activation_gradient1 = activation_gradients[gradients[0]]
        self.activation2 = activation_functions[activations[1]]
        self.activation_gradient2 = activation_gradients[gradients[1]]
        self.candidate_mem_weight = None
        self.input_weight = None
        self.forget_weight = None
        self.output_weight = None
        self.candidate_mem_weight_bias = None
        self.input_weight_bias = None
        self.forget_weight_bias = None
        self.output_weight_bias = None
        self.mem_cells = mem_cells
        self.bp_time_steps = bp_time_steps
        self.hidden_input_concat = None
        self.input_shape = input_shape

    def initialize_layer(self, optimizer):
        concatenation_length = self.input_shape[1] + self.mem_cells
        # Xavier initialization for RNN structures
        limit = np.sqrt(6.0 / (concatenation_length + self.mem_cells))

        self.candidate_mem_weight = np.random.uniform(-limit, limit, (self.mem_cells, concatenation_length))
        self.input_weight = np.random.uniform(-limit, limit, (self.mem_cells, concatenation_length))
        self.output_weight = np.random.uniform(-limit, limit, (self.mem_cells, concatenation_length))
        self.forget_weight = np.random.uniform(-limit, limit, (self.mem_cells, concatenation_length))

        limit = np.sqrt(6.0 / (self.input_shape[1] + self.mem_cells))
        # Layer output refers to the output projection of hidden state (normally a dense layer would fill this role in most architectures)
        self.layer_output_weight = np.random.uniform(-limit, limit, (self.input_shape[1], self.mem_cells))

        self.candidate_mem_weight_bias = np.zeros(self.mem_cells)
        self.input_weight_bias = np.zeros(self.mem_cells)
        self.output_weight_bias = np.zeros(self.mem_cells)
        self.forget_weight_bias = np.ones(self.mem_cells) # Common trick to improve training stability
        self.layer_output_weight_bias = np.zeros(self.input_shape[1])

        self.candidate_mem_weight_optimizer = copy.copy(optimizer)
        self.input_weight_optimizer = copy.copy(optimizer)
        self.output_weight_optimizer = copy.copy(optimizer)
        self.forget_weight_optimizer = copy.copy(optimizer)
        self.layer_output_weight_optimizer = copy.copy(optimizer)

        self.candidate_mem_weight_bias_optimizer = copy.copy(optimizer)
        self.input_weight_bias_optimizer = copy.copy(optimizer)
        self.output_weight_bias_optimizer = copy.copy(optimizer)
        self.forget_weight_bias_optimizer = copy.copy(optimizer)
        self.layer_output_weight_bias_optimizer = copy.copy(optimizer)

    def forward_pass(self, input, training = True):
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape
        self.input_dim = input_dim
        self.states = np.zeros((batch_size, timesteps + 1, self.mem_cells))
        self.hidden_states = np.zeros((batch_size, timesteps + 1, self.mem_cells))
        self.layer_output = np.zeros((batch_size, timesteps, self.input_shape[1]))
        self.inputs = np.zeros((batch_size, timesteps, self.mem_cells))
        self.outputs = np.zeros((batch_size, timesteps, self.mem_cells))
        self.candidate_mems = np.zeros((batch_size, timesteps, self.mem_cells))
        self.forgets = np.zeros((batch_size, timesteps, self.mem_cells))

        # Set last time step (t = -1) to zeros in order to prevent loop from crashing at t = 0
        self.hidden_states[:, -1] = np.zeros((batch_size, self.mem_cells))
        self.states[:, -1] = np.zeros((batch_size, self.mem_cells))
        self.hidden_input_concat = np.zeros((batch_size, timesteps, self.mem_cells + self.input_dim))

        for t in range(timesteps):
            self.hidden_input_concat[:, t] = np.hstack((self.layer_input[:, t], self.hidden_states[:, t - 1]))

            candidate_mem_activation = self.candidate_mem_weight.dot(self.hidden_input_concat[:, t].T).T + self.candidate_mem_weight_bias
            input_activation = self.input_weight.dot(self.hidden_input_concat[:, t].T).T + self.input_weight_bias
            forget_activation = self.forget_weight.dot(self.hidden_input_concat[:, t].T).T + self.forget_weight_bias
            output_activation = self.output_weight.dot(self.hidden_input_concat[:, t].T).T + self.output_weight_bias

            self.candidate_mems[:, t] = self.activation1(candidate_mem_activation)
            self.inputs[:, t] = self.activation2(input_activation)
            self.forgets[:, t] = self.activation2(forget_activation)
            self.outputs[:, t] = self.activation2(output_activation)
            self.states[:, t] = self.forgets[:, t] * self.states[:, t - 1] + self.inputs[:, t] * self.candidate_mems[:, t]
            self.hidden_states[:, t] = self.outputs[:, t] * self.activation1(self.states[:, t])
            self.layer_output[:, t] = self.hidden_states[:, t].dot(self.layer_output_weight.T) + self.layer_output_weight_bias
            
        return self.layer_output
        
    def backward_pass(self, accumulated_grad_hidden):
        batch_size, timesteps, _ = accumulated_grad_hidden.shape

        weight_input_grad = np.zeros_like(self.input_weight)
        weight_output_grad = np.zeros_like(self.output_weight)
        weight_layer_output_grad = np.zeros_like(self.layer_output_weight)
        weight_forget_grad = np.zeros_like(self.forget_weight)
        weight_candidate_mems_grad = np.zeros_like(self.candidate_mem_weight)

        input_weight_bias_grad = np.zeros_like(self.input_weight_bias)
        output_weight_bias_grad = np.zeros_like(self.output_weight_bias)
        layer_output_weight_bias_grad = np.zeros_like(self.layer_output_weight_bias)
        forget_weight_bias_grad = np.zeros_like(self.forget_weight_bias)
        candidate_mem_weight_bias_grad = np.zeros_like(self.candidate_mem_weight_bias)

        hidden_input_concat_grad = np.zeros_like(self.hidden_input_concat)
        accumulated_grad_hidden_next = np.zeros((batch_size, timesteps, self.input_shape[1]))

        for t in reversed(range(timesteps)):
            weight_layer_output_grad += accumulated_grad_hidden[:, t].T.dot(self.hidden_states[:, t])
            accumulated_grad_hidden_output = accumulated_grad_hidden[:, t].dot(self.layer_output_weight)
            weight_state_diff = (self.outputs[:, t] * accumulated_grad_hidden_output * 
                                 self.activation_gradient1(self.states[:, t]))

            for t2 in reversed(np.arange(max(0, t - self.bp_time_steps), t + 1)):
                weight_input_diff = self.candidate_mems[:, t2] * weight_state_diff
                weight_output_diff = self.activation1(self.states[:, t2]) * accumulated_grad_hidden_output
                weight_forget_diff = self.states[:, t2 - 1] * weight_state_diff
                weight_candidate_mems_diff = self.inputs[:, t2] * weight_state_diff

                # Chain rule gets applied here (hence the naming scheme)
                weight_input_diff_chain = weight_input_diff * self.activation_gradient2(self.inputs[:, t2])
                weight_output_diff_chain = weight_output_diff * self.activation_gradient2(self.outputs[:, t2])
                weight_forget_diff_chain = weight_forget_diff * self.activation_gradient2(self.forgets[:, t2])
                weight_candidate_mems_diff_chain = weight_candidate_mems_diff * self.activation_gradient1(self.candidate_mems[:, t2])

                weight_input_grad += weight_input_diff_chain.T.dot(self.hidden_input_concat[:, t2])
                weight_output_grad += weight_output_diff_chain.T.dot(self.hidden_input_concat[:, t2])
                weight_forget_grad += weight_forget_diff_chain.T.dot(self.hidden_input_concat[:, t2])
                weight_candidate_mems_grad += weight_candidate_mems_diff_chain.T.dot(self.hidden_input_concat[:, t2])

                input_weight_bias_grad += np.sum(weight_input_diff_chain, axis = 0)
                output_weight_bias_grad += np.sum(weight_output_diff_chain, axis = 0)
                layer_output_weight_bias_grad += np.sum(accumulated_grad_hidden[:, t], axis = 0)
                forget_weight_bias_grad += np.sum(weight_forget_diff_chain, axis = 0)
                candidate_mem_weight_bias_grad += np.sum(weight_candidate_mems_diff_chain, axis = 0)

                hidden_input_concat_grad[:, t2] += weight_input_diff_chain.dot(self.input_weight)
                hidden_input_concat_grad[:, t2] += weight_output_diff_chain.dot(self.output_weight)
                hidden_input_concat_grad[:, t2] += weight_forget_diff_chain.dot(self.forget_weight)
                hidden_input_concat_grad[:, t2] += weight_candidate_mems_diff_chain.dot(self.candidate_mem_weight)

                accumulated_grad_hidden_next[:, t2] += hidden_input_concat_grad[:, t2, :self.input_dim]
                grad_hidden_previous = hidden_input_concat_grad[:, t2, self.input_dim:]

                weight_state_diff = (self.outputs[:, t2 - 1] * grad_hidden_previous * 
                                    self.activation_gradient1(self.states[:, t2 - 1]) + 
                                    weight_state_diff * self.forgets[:, t2])

        # Limit gradient norms (alleviates exploding gradients)
        weight_input_grad = gradient_clip(weight_input_grad, max_norm = 0.25)
        weight_output_grad = gradient_clip(weight_output_grad, max_norm = 0.25)
        weight_layer_output_grad = gradient_clip(weight_layer_output_grad, max_norm = 0.25)
        weight_forget_grad = gradient_clip(weight_forget_grad, max_norm = 0.25)
        weight_candidate_mems_grad = gradient_clip(weight_candidate_mems_grad, max_norm = 0.25)

        input_weight_bias_grad = gradient_clip(input_weight_bias_grad, max_norm = 0.25)
        output_weight_bias_grad = gradient_clip(output_weight_bias_grad, max_norm = 0.25)
        layer_output_weight_bias_grad = gradient_clip(layer_output_weight_bias_grad, max_norm = 0.25)
        forget_weight_bias_grad = gradient_clip(forget_weight_bias_grad, max_norm = 0.25)
        candidate_mem_weight_bias_grad = gradient_clip(candidate_mem_weight_bias_grad, max_norm = 0.25)

        self.input_weight = self.input_weight_optimizer.update(self.input_weight, weight_input_grad)
        self.output_weight = self.output_weight_optimizer.update(self.output_weight, weight_output_grad)
        self.layer_output_weight = self.layer_output_weight_optimizer.update(self.layer_output_weight, weight_layer_output_grad)
        self.forget_weight = self.forget_weight_optimizer.update(self.forget_weight, weight_forget_grad)
        self.candidate_mem_weight = self.candidate_mem_weight_optimizer.update(self.candidate_mem_weight, weight_candidate_mems_grad)

        self.input_weight_bias = self.input_weight_bias_optimizer.update(self.input_weight_bias, input_weight_bias_grad)
        self.output_weight_bias = self.output_weight_bias_optimizer.update(self.output_weight_bias, output_weight_bias_grad)
        self.layer_output_weight_bias = self.layer_output_weight_bias_optimizer.update(self.layer_output_weight_bias, layer_output_weight_bias_grad)
        self.forget_weight_bias = self.forget_weight_bias_optimizer.update(self.forget_weight_bias, forget_weight_bias_grad)
        self.candidate_mem_weight_bias = self.candidate_mem_weight_bias_optimizer.update(self.candidate_mem_weight_bias, candidate_mem_weight_bias_grad)

        return accumulated_grad_hidden_next
    
    def parameters(self):
        return (np.prod(self.input_weight.shape) + np.prod(self.output_weight.shape) + 
                np.prod(self.forget_weight.shape) + np.prod(self.candidate_mem_weight.shape) +
                np.prod(self.input_weight_bias.shape) + np.prod(self.output_weight_bias.shape) + 
                np.prod(self.forget_weight_bias.shape) + np.prod(self.candidate_mem_weight_bias.shape))
    
    def get_output_shape(self):
        return self.input_shape      

# 2D Convolution Layer
class Conv2D(Layer):

    def __init__(self, filters, filter_shape, input_shape = None, padding_type = "same", stride = 1):
        self.filters = filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.padding_type = padding_type
        self.stride = stride
        self.trainable = True
    
    def initialize_layer(self, optimizer):
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]

        limit = 1 / math.sqrt(math.prod(self.filter_shape))
        self.weight = np.random.uniform(-limit, limit, size = (self.filters, channels, filter_height, filter_width))
        self.weight_bias = np.zeros((self.filters, 1))

        self.weight_optimizer = copy.copy(optimizer)
        self.weight_bias_optimizer = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight_bias.shape)
    
    # Logic for forward and backward pass is similiar to that of Dense layer
    def forward_pass(self, X, training = True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X

        # Turn image shape into column shape and vice versa for weights
        self.X_col = image_to_column(X, self.filter_shape, self.stride, self.padding_type)

        # Convert weight to column shape
        self.weight_col = self.weight.reshape((self.filters, -1))

        output = self.weight_col.dot(self.X_col) + self.weight_bias
        output = output.reshape(self.get_output_shape() + (batch_size, ))

        # Make batch_size come first
        return output.transpose(3, 0, 1, 2)
    
    def backward_pass(self, accum_grad):
        # Convert gradient into column shape
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape((self.filters, -1))

        if self.trainable:
            grad_weight = accum_grad.dot(self.X_col.T).reshape(self.weight.shape)
            grad_weight_bias = np.sum(accum_grad, axis = 1, keepdims = True)

            self.weight = self.weight_optimizer.update(self.weight, grad_weight)
            self.weight_bias = self.weight_bias_optimizer.update(self.weight_bias, grad_weight_bias)

        accum_grad = self.weight_col.T.dot(accum_grad)
        accum_grad = column_to_image(accum_grad, self.layer_input.shape, self.filter_shape, 
                                     stride = self.stride, 
                                     padding_type = self.padding_type)
        
        return accum_grad
        

    def get_output_shape(self):
        _, height, width = self.input_shape
        pad_height, pad_width = determine_padding(self.filter_shape, self.padding_type)

        output_height = int((height + np.sum(pad_height) - self.filter_shape[0]) / self.stride + 1)
        output_width = int((width + np.sum(pad_width) - self.filter_shape[1]) / self.stride + 1)

        return self.filters, output_height, output_width
    
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

    # Uses equation referenced in padding function
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
    i = i1.reshape(-1, 1) + i2.reshape(1, -1)
    j = j1.reshape(-1, 1) + j2.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)

# Used in forward pass
# Converts image data into column shape
def image_to_column(images, filter_shape, stride, padding_type = "same"):
    pad_height, pad_width = determine_padding(filter_shape, padding_type)

    images_padded = np.pad(images, ((0, 0), (0, 0), pad_height, pad_width), mode = "constant")

    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_height, pad_width), stride)

    # Create columns that contain each convolution window and transpose them for quick computation (GeMM)
    # Reference: https://spatial-lang.org/gemm/

    cols = images_padded[:, k, i, j]
    channels = images.shape[1]

    # A transport argument of -1 means that the row or column is automatically assigned based on original shape
    # i.e rows = size / columns and vice versa
    cols = cols.transpose(1, 2, 0).reshape(np.prod(filter_shape) * channels, -1)
    return cols

# Used in backward pass
# Converts column shaped input into image shape
# Note: This is not intended to recreate the original image data, but serve as the adjoint operation to image_to_column
# i.e this only serves as the backward operation in terms of backpropagation
def column_to_image(cols, images_shape, filter_shape, stride, padding_type = "same"):
    batch_size, channels, height, width = images_shape
    pad_height, pad_width = determine_padding(filter_shape, padding_type)

    images_padded = np.zeros((batch_size, channels, np.sum(pad_height) + height, np.sum(pad_width) + width))

    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_height, pad_width), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)

    # This will sum up overlapping pixels in the convolution columns as their gradient contributions to each convolution patch would sum up
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    return images_padded[:, :, pad_height[0] : height + pad_height[0], pad_width[0] : width + pad_width[0]]
        
# Layer with only activation functions
class Activation(Layer):

    def __init__(self, name):
        self.activation_function = activation_functions[name]
        self.activation_name = name
        self.trainable = True   
    
    def get_layer_name(self):
        return "(%s) Activation" % (self.activation_name)
    
    def forward_pass(self, X, training = True):
        self.layer_input = X
        return self.activation_function(X)
    
    def backward_pass(self, accumulated_grad):
        if self.activation_name == 'softmax': 
            softmax_output = self.activation_function(self.layer_input)
            return activation_gradients['softmax_gradient'](accumulated_grad, softmax_output)
        return accumulated_grad * activation_gradients[self.activation_name + '_gradient'](self.layer_input)
    
    def get_output_shape(self):
        return self.input_shape

# Layer that sets a certain percentage of previous output values to 0
class Dropout(Layer):
    
    def __init__(self, probability = 0.2):
        self.probability = probability
        self.mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    # We want to keep the expected value of neurons if not training as the weights were adjusted with the dropout in mind
    # i.e we want to mimick what happened during training if not training through expected value
    def forward_pass(self, X, training = True):
        temp_mask = (1 - self.probability)
        if training:
            self.mask = np.random.uniform(size = X.shape) > self.probability
            temp_mask = self.mask
        return temp_mask * X
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.mask

    def parameters(self):
        return 0
    
    def get_output_shape(self):
        return self.input_shape

# Normalizes activation output for a specific feature across a batch (rather than a specific sample as with layer normalization)
class BatchNormalization(Layer):

    def __init__(self, momentum = 0.99):
        self.momentum = momentum
        self.trainable = True
        self.epsilon = 0.01
        self.running_mean = None
        self.running_var = None

    def initialize_layer(self, optimizer):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)
    
    def forward_pass(self, X, training = True):
        # Initialize mean on first run
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis = 0)
            self.running_var = np.var(X, axis = 0)
        
        if training and self.trainable:
            mean = np.mean(X, axis = 0)
            var = np.var(X, axis = 0)
            # Exponential running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        
        else:
            # Use most recent running_mean
            mean = self.running_mean
            var = self.running_var
        
        # For backward pass
        self.X_centered = X - mean
        self.inverse_standard_deviation = 1 / np.sqrt(var + self.epsilon)

        X_normalized = self.X_centered * self.inverse_standard_deviation
        output = self.gamma * X_normalized + self.beta

        return output

    def backward_pass(self, accum_grad):
        # Save values from forward pass
        prev_gamma = self.gamma

        # Update weights if trainable
        if self.trainable:
            X_normalized = self.X_centered * self.inverse_standard_deviation
            grad_gamma = np.sum(accum_grad * X_normalized, axis = 0)
            grad_beta = np.sum(accum_grad, axis = 0)

            self.gamma = self.gamma_optimizer.update(self.gamma, grad_gamma)
            self.beta = self.beta_optimizer.update(self.beta, grad_beta)
        
        batch_size = accum_grad.shape[0]

        # The gradient of loss with respect to layer inputs
        accum_grad = (1 / batch_size) * (prev_gamma * self.inverse_standard_deviation * batch_size * accum_grad - 
                                        np.sum(accum_grad, axis = 0) - self.X_centered * self.inverse_standard_deviation ** 2
                                        * np.sum(accum_grad * self.X_centered, axis = 0))
        
        return accum_grad
    
    def get_output_shape(self):
        return self.input_shape
    
# Like batch normalization but for a sample
class LayerNormalization(Layer):

    def __init__(self, input_size = None):
        self.trainable = True
        self.epsilon = 0.01
        self.prev_mean = None
        self.prev_var = None
        self.input_shape = (input_size, )

    def initialize_layer(self, optimizer):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self, X, training = True):
        if training and self.trainable:
            mean = np.mean(X, axis = -1, keepdims = True)
            var = np.var(X, axis = -1, keepdims = True)
            self.prev_mean = mean
            self.prev_var = var
        else:
            mean = self.prev_mean
            var = self.prev_var
            # Ensures that mean and variance are broadcastable to X
            if mean is not None and X.shape[-1] == mean.shape[-1]:
                mean = np.broadcast_to(mean, X.shape)
                var = np.broadcast_to(var, X.shape) 
            
        # For backward pass
        self.X_centered = X - mean
        self.inverse_standard_deviation = 1 / np.sqrt(var + self.epsilon)

        X_normalized = self.X_centered * self.inverse_standard_deviation
        output = self.gamma * X_normalized + self.beta

        return output

    def backward_pass(self, accum_grad):
        # Save values from forward pass
        prev_gamma = self.gamma

        # Update weights if trainable
        if self.trainable:
            X_normalized = self.X_centered * self.inverse_standard_deviation
            reduction_axes = tuple(range(accum_grad.ndim - 1))
            grad_gamma = np.sum(accum_grad * X_normalized, axis = reduction_axes)
            grad_beta = np.sum(accum_grad, axis = reduction_axes)

            self.gamma = self.gamma_optimizer.update(self.gamma, grad_gamma)
            self.beta = self.beta_optimizer.update(self.beta, grad_beta)
        
        feature_length = accum_grad.shape[-1]

        grad_normalized = accum_grad * prev_gamma

        # The gradient of loss with respect to layer inputs
        accum_grad = ((self.inverse_standard_deviation / feature_length) * (feature_length * grad_normalized - 
                     np.sum(grad_normalized, axis = -1, keepdims = True) - (self.X_centered * self.inverse_standard_deviation ** 2
                     * np.sum(grad_normalized * self.X_centered, axis = -1, keepdims = True))))
        
        return accum_grad

    def get_output_shape(self):
        return self.input_shape

# Flattens multidimesional matrix into 2-D matrix
class Flatten(Layer):

    # Input shape of only one batch/sample (i.e no batch dimension)
    def __init__(self, batch_input_shape = None):
        self.previous_shape = None
        self.trainable = True
        self.input_shape = batch_input_shape
    
    def forward_pass(self, X, training = True):
        self.previous_shape = X.shape
        return X.reshape((X.shape[0], -1))
    
    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.previous_shape)

    def parameters(self):
        return 0
    
    # Note: This is the output shape of one specific sample (the -1 dimesion is the sample dimesion in the forward pass)
    def get_output_shape(self):
        return (np.prod(self.input_shape),)

        


        





        
        
        



    
    