import numpy as np 
from DecisionTree import DecisionTree
from collections import Counter


class RandomForest : 
    def __init__(self,n_trees=10, max_depth=100, min_samples_split=2, n_features=None):  
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y) :
        for _ in range(self.n_trees) :
            tree = DecisionTree(self.min_samples_split, 
                         self.max_depth, 
                         self.n_features)

            # we randomly select a population from X and fit it to each decision tree
            X_sample,y_sample = self._get_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _get_sample(self, X, y):
        n_samples= X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        # Returns the label that appears most often in y
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        """ say we have 9 decision trees in the forest, that means there are 9 arrays , each has the predictions of one tree
        y_hats -> [[1,0,1],[1,2,1],[3,3,2],[0,0,1],[1,1,0],[1,0,0],[2,1,2],[2,1,2],[3,2,1]]
        """
        # if classification we select the most common, else we select the mean or average
        trees_preds = np.swapaxes(predictions, 0,1)
        predictions = np.array([ self._most_common_label(pred) for pred in trees_preds ])
        return predictions