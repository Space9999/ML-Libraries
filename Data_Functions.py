import numpy as np

class Data_Functions():
    def calculate_covariance_matrix(X, Y = None):
        if Y is None:
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (X - X.mean(axis = 0)).T.dot(Y - Y.mean(axis = 0)) / (n_samples - 1)

        return np.array(covariance_matrix, dtype = float)
    
    def shuffle(X, y, seed = None):
        if seed:
            np.random.seed(seed)
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        return X[index], y[index]
    
    def train_test_split(X, y, test_size = 0.5, is_shuffled = True, seed = None):
        if is_shuffled:
            X, y = Data_Functions.shuffle(X, y, seed)
        index_split = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:index_split], X[index_split:]
        y_train, y_test = y[:index_split], y[index_split:]

        return X_train, X_test, y_train, y_test
    
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis = 0) / len(y_true)
        return accuracy
    
    def normalize(X, axis = -1, order = 2):
        l2_norm = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2_norm[l2_norm == 0] = 1
        return X / np.expand_dims(l2_norm, axis)
    
    def to_categorical(x, n_col = None):
        if not n_col:
            n_col = np.amax(x) + 1
        one_hot = np.zeros((x.shape[0], n_col))
        one_hot[np.arange(x.shape[0]), x] = 1
        return one_hot
