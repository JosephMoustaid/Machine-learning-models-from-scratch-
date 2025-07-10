from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree 

data = datasets.load_breast_cancer()

X,y = data.data , data.target  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42)

dt_clf = DecisionTree()
dt_clf.fit(X_train, y_train)

y_hat = dt_clf.predict(X_test)



def accuracy(y_test, y_hat):
    return np.sum(y_hat == y_test) / len(y_test)

acc = accuracy(y_test, y_hat)
print(f"the accuracy of this model is : {np.round(accuracy(y_test, y_hat),2)}")

