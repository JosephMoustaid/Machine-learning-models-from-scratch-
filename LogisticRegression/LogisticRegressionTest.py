import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# lets use the breast cancer data set from sklern
bc = datasets.load_breast_cancer()

X, y= bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)


lr_clf = LogisticRegression(lr=0.01)
lr_clf.fit(X_train, y_train)
y_hat = lr_clf.predict(X_test)

def accuracy(y_test, y_hat):
    return np.sum(y_hat == y_test) / len(y_test)

acc = accuracy(y_test, y_hat)
print(f"the accuracy of this model is : {np.round(accuracy(y_test, y_hat),2)}")

