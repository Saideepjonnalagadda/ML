import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

# Load Data
iris = load_iris()
X = iris.data[:, 2:4]  # Only Petal length and width
y = (iris.target != 0).astype(int)

# Split Data (using the same split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the model
model = LogisticRegression()
model.load('model_classifier1.npz')

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy for Petal Features: {accuracy}")
