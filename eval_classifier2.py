import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = load_iris()
X = iris.data[:, 0:2]
y = (iris.target != 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression()
model.load('classification2.npz')
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy for classification 2 using sepal features: {accuracy}")
