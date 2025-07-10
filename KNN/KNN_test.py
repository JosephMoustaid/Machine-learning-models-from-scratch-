import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from KNN import KNN

# we get some colors for the visualization
color_map = ListedColormap(['#FF0000' , '#00FF00' , '#0000FF'])

# we load the iris dataset
iris= datasets.load_iris()
X , y = iris.data , iris.target
# train test split the dataset
X_train, X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1234)

# visualize the dataset
"""
plt.figure()
plt.scatter(X[:,2] , X[:,3] , c=y , cmap=color_map , edgecolor='k' , s=20)
plt.show()
"""

clt = KNN(k=5)
clt.fit(X_train , y_train)
predictions = clt.predict(X_test)
print(predictions)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("The Accuracy is : " ,  accuracy)


