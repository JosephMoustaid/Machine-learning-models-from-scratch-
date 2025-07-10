import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: if all labels are same, max depth reached, or not enough samples
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best feature and threshold to split on
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes recursively
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate the information gain for each threshold
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # --- Information Gain ---
        # IG = Entropy(parent) - [weighted average of Entropy(children)]
        
        # 1. Compute parent entropy
        parent_entropy = self._entropy(y)

        # 2. Split data
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # 3. Compute weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        # e_l = Entropy of left branch
        # e_r = Entropy of right branch
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # Weighted child entropy
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # 4. IG = parent - weighted children
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        # Returns the indices of the rows where the value is <= or > the threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        --- Entropy formula ---
        Entropy = -∑(P(c) * log(P(c))) 
        where P(c) is the probability of class c in current node.

        Example: if y = [1, 1, 0, 0], then:
        - Class 0: 2 occurrences → P(0) = 2/4 = 0.5
        - Class 1: 2 occurrences → P(1) = 0.5
        Entropy = -[0.5*log(0.5) + 0.5*log(0.5)] = 0.693...
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        # Returns the label that appears most often in y
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        # Traverse the tree for each input sample
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
