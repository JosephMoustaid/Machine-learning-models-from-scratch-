import numpy as np

class LinearRegression:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # here , we give weights and array of zero , because the user might input multiple features 
        # and we need to have a weight for each feature, initially we set all weights to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def MSE(self, y_test , y_pred):
        # dot(A,Z) -> does W * Z   . BTW both are arrays
        N =  len(y_test)
        return 1/N * np.dot( (y_test - y_pred), (y_test - y-pred) )





    def MSE(self, y_test, y_pred):
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.

        Returns:
        float: The Mean Squared Error.
        """
        n_samples = len(y_test)
        return (1 / n_samples) * np.dot((y_test - y_pred), (y_test - y_pred))