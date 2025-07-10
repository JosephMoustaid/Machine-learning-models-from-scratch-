import math

import numpy as np 
# we will sue numpy to sort the distnace array 
# so we get the first k elements , the smallest distances

from collections import Counter
# we will use counter to get the most common votes of the k-neighbours 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self , k=3 ):
        self.k = k 

    def fit(self, X , y):
        self.X_train = X
        self.y_train = y
        
    def predict(self , X):
        return [int(self._predict(x)) for x in X]
        
    # Helper function
    def _predict(self , X):
        # compute the distnaces 
        distances =  [euclidean_distance(x_train , X) for x_train in self.X_train] ;
        
        # get the closest k neighbours
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[indice] for indice in k_indices]
        
        # get the majority vote 
        most_common = Counter(k_nearest_labels).most_common(1)  # Take the most common label
        return most_common[0][0]  # Return the label, not the tuple , because Counter returns the common label and all it's occurances in the table9        


    