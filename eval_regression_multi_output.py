import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.data[:, [2, 3]]  # Predict both petal length and petal width

# Select input features (Use sepal length and sepal width to predict both petal length and width)
X_features = X[:, [0, 1]]  # Use sepal length and sepal width

# Split data into training and testing sets (consistent separation)
_, X_test, _, y_test = train_test_split(X_features, y, test_size=0.1, random_state=42)

# Load the multi-output model
model = LinearRegression()
model.load('model_regression_multi_output.npz')

# Evaluate the performance on the testing data
test_mse = model.score(X_test, y_test)
print(f"Test MSE for Multi-Output Regression: {test_mse}")
