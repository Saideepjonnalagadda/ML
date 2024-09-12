import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load data
iris = load_iris()
X = iris.data[:, :2]  # Use sepal length and width as features
y = iris.data[:, 3]   # Predict petal width

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the model
model = LinearRegression()
model.load('model_regression1.npz')

# Evaluate the model
test_mse = model.score(X_test, y_test)
print(f"Test MSE for Regression Model 1: {test_mse}")
