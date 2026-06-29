import numpy as np

class Base_Neural_Network():

    # Validation data must be in a tuple format (X set, y set)
    def __init__(self, optimizer, loss, loss_grad, validation_data = None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss
        self.loss_gradient = loss_grad
        self.parameters = 0

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X" : X, "y" : y}
    
    # Method used to freeze layer weights if needed
    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        # If the layer added is not the 1st one, then set the shape of the input to that of the output of last layer added
        if self.layers:
            layer.set_input_shape(self.layers[-1].get_output_shape())

        # If the layer needs to be initialized, then initalize layer with appropriate optimizer and weights
        if hasattr(layer, 'initialize_layer'):
            layer.initialize_layer(optimizer = self.optimizer)
        
        self.parameters += layer.parameters()
        self.layers.append(layer)

    def test_batch(self, X, y):
        y_pred = self.forward_pass(X)
        loss = self.loss_function(y, y_pred)

        return loss
    
    def train_batch(self, X, y):
        y_pred = self.forward_pass(X)
        loss = self.loss_function(y, y_pred)
        loss_grad = self.loss_gradient(y, y_pred)
        self.backward_pass(loss_grad)

        return loss
    
    def fit(self, X, y, epochs, batch_size):
        training_loss = []
        val_loss = []
        for i in range(epochs):
            batch_loss = []
            for j in range(0, len(X), batch_size):
                X_batch = X[j : j + batch_size]
                y_batch = y[j : j + batch_size]
                loss = self.train_batch(X_batch, y_batch)
                batch_loss.append(np.mean(loss))

            training_loss.append(np.mean(batch_loss))

            if self.val_set is not None:
                val_loss.append(self.test_batch(self.val_set["X"], self.val_set["y"]))
        
        return training_loss, val_loss
    
    def forward_pass(self, X, training = True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output
    
    def get_parameters(self):
        return self.parameters
    
    def backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
    
    def predict(self, X):
        return self.forward_pass(X, training = False)

                



    
