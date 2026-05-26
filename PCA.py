import numpy as np
import Data_Functions as data_func

# Principal Component Analysis
# Method for reducing dimensionality of given feature space and maximizing variance on each feature axis
class PCA():
    def transform(X, n_components):
        X_centered = X - np.mean(X, axis=0)
        covariance_matrix = data_func.Data_Functions.calculate_covariance_matrix(X_centered)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and eigenvectors by largest to smallest and get the first n_components
        index = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[index][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, index])[:, :n_components]

        # Project data onto components (the given eigenvectors)
        X_transformed = X.dot(eigenvectors)

        return X_transformed