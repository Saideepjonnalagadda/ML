import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load data
iris = load_iris()
X = iris.data[:, [1, 2]]  # Use sepal width and petal length as features
y = iris.data[:, 3]       # Predict petal width

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, max_epochs=100, patience=3)

# Save the model
model.save('model_regression3.npz')

# Plot the loss function
plt.plot(model.loss_history)
plt.title('Training Loss over Epochs (Regression 3)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss on Validation Set')
plt.show()

print("Training complete and model saved.")
