from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import your custom LinearRegression class
from LinearRegression import LinearRegression

class LinearRegressionTest:
    def __init__(self):
        # Load the iris dataset
        iris = load_iris()
        
        # Convert to DataFrame
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Select one feature for regression
        X = df[["sepal length (cm)"]]  # Ensure it's a DataFrame
        y = df["target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # Test the LinearRegression class
        self.reg = LinearRegression()
        self.reg.fit(X_train, y_train)
        self.predictions = self.reg.predict(X_test)
        self.mse = self.reg.MSE(y_test, self.predictions)    

        print("MSE:", self.mse)

        # Visualizing regression model
        sns.set_theme(color_codes=True)
        plt.figure(figsize=(8, 6))
        sns.regplot(x=X_test.iloc[:, 0], y=y_test, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Target")
        plt.title("Linear Regression on Sepal Length vs. Target")
        plt.show()

if __name__ == "__main__":
    test = LinearRegressionTest()





