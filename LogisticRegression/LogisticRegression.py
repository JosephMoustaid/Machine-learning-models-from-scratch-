import numpy as np

# The sygmoid function
def sigmoid (x) :
    return 1 / (1 + np.exp(-x))

class LogisticRegression :
# lr -> learning rate: how fast gradient descent works
# n_iter -> number of iteration 
    def __init__(self, lr=0.001, n_iter=100):
        self.lr = lr
        self.n_iter=n_iter
        self.weights = None
        self.bias = None

    
    def fit(self, X, y): 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0
        for i in range(0, self.n_iter) :
            # first calculate the y = wx+ b then pass those to the sygmoid
            linear_predictions = np.dot(X,self.weights) + self.bias
            predictions = sigmoid(linear_predictions)
            
            # calulate the gradient descent (grtadient of the wights and gradient of the bias)
            dw =  (1/n_samples) * np.dot(X.T, (predictions - y)) # T is the transpose of X
            db =  (1/n_samples) * np.sum(predictions - y) 
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
            

    def predict(self, X):
        linear_predictions = np.dot(X,self.weights) + self.bias
        y_hat = sigmoid(linear_predictions)
        # since y_hat contains the probabilities of each class we need to select 1 or 0 (y_hat contains values between 0 and 1 for each probablility due to the sygmoid function)
        class_predictions = [0 if y<=0.5 else 1 for y in y_hat]    
        return class_predictions
            
