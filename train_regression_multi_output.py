import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.data[:, [2, 3]]  # Predict both petal length and petal width

# Select input features (Use sepal length and sepal width to predict both petal length and width)
X_features = X[:, [0, 1]]  # Use sepal length and sepal width

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.1, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, max_epochs=100, patience=3)

# Save the model parameters
model.save('model_regression_multi_output.npz')

# You can also generate loss plots here if required.

print("Training multi-output model complete! Model parameters saved.")
