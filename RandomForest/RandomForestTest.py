from RandomForest import RandomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 

df = datasets.load_breast_cancer()
X, y= df.data, df.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3
)

def accuracy(y_test, y_hat) :
    accuracy = np.sum(y_hat == y_test) / len(y_test)
    return accuracy

rf_clf = RandomForest()
rf_clf.fit(X_train, y_train)
y_hat = rf_clf.predict(X_test)

print(f"The accuracy of the random forest classifier is {accuracy(y_test, y_hat)}" )



