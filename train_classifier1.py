import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from LogisticRegression import LogisticRegression

# Load Data
iris = load_iris()
X = iris.data[:, 2:4]  # Only Petal length and width
y = (iris.target != 0).astype(int)  # Binary classification (setosa or not setosa)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(learning_rate=0.01, max_epochs=1000)
model.fit(X_train, y_train)

# Save the model
model.save('model_classifier1.npz')

# Visualize Decision Boundary
plot_decision_regions(X_train, y_train, clf=model)
plt.title("Decision boundary for Logistic Regression using Petal Features")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

print("Training complete! Model parameters saved.")
