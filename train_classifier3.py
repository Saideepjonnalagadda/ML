import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
iris = load_iris()
X = iris.data
y = (iris.target != 0).astype(int)#Binary classification-setosa or not setosa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression(learning_rate=0.01, max_epochs=1000)
model.fit(X_train, y_train)
model.save('classification3.npz')
print("Model trained, parameters saved.")
