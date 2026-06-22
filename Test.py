import Base_Neural_Network as NN
import Optimizers
import Loss_Functions
from Layers import Dense, Activation
import Data_Functions as data_func
import PCA as pca_import

from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt

data = datasets.load_digits()
digit_target_1 = 3
digit_target_2 = 9
index = np.append(np.where(data.target == digit_target_1)[0], np.where(data.target == digit_target_2)[0])
y = data.target[index]

# Change labels to 0 and 1
y[y == digit_target_1] = 0
y[y == digit_target_2] = 1

X = data.data[index]

X = pca_import.PCA.transform(X, n_components = 10) # Reduce to 10 dimesions
n_samples, n_features = np.shape(X)

X_train, X_test, y_train, y_test = data_func.Data_Functions.train_test_split(X, y, test_size = 0.5)

mlp = NN.Base_Neural_Network(optimizer = Optimizers.Adam(), loss = Loss_Functions.categorical_cross_entropy, loss_grad = Loss_Functions.categorical_cross_entropy_grad)
mlp.add(Dense(input_shape = (n_features, ), n_units = 64))
mlp.add(Activation('relu'))
mlp.add(Dense(n_units = 64))
mlp.add(Activation('relu'))
mlp.add(Dense(n_units = 2))
mlp.add(Activation('softmax'))

mlp.fit(X_train, data_func.Data_Functions.to_categorical(y_train), epochs = 250, batch_size = 50)

y_pred = np.argmax(mlp.predict(X_test), axis = 1)

print("Accuracy: " + str(data_func.Data_Functions.accuracy_score(y_test, y_pred)))

plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
plt.ylabel("Feature 2")
plt.xlabel("Feature 1")
plt.title("Digit dataset (digits %s and %s)" % (digit_target_1, digit_target_2))
plt.show()
