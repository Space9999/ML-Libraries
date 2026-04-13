import Base_Neural_Network as NN
import Optimizers
import Loss_Functions
import numpy as np
import Layers

# Just a quick test of the loss functions and the overall structure
X = np.array([4, 5, 8, 9, 10, 20])
Y = np.array([8, 10, 16, 18, 20, 40])
network = NN.Base_Neural_Network(Optimizers.Simplified_SGD, Loss_Functions.mse, Loss_Functions.mse_grad, None)
print(network.train_batch(X, Y))
