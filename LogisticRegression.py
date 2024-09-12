import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000, regularization=0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.regularization = regularization
        self.weights = None

    # Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Binary Cross-Entropy Loss
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15  # Add a small epsilon to avoid division by 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        np.random.seed(42)
        
        n_samples, n_features = X.shape
        X = np.hstack([X, np.ones((n_samples, 1))])  # Add a bias column
        self.weights = np.random.randn(n_features + 1, 1)  # +1 for the bias term
        
        for epoch in range(self.max_epochs):
            # Calculate predictions
            logits = X @ self.weights
            y_pred = self.sigmoid(logits)
            
            # Compute the error term (difference between predictions and true values)
            error = y_pred - y.reshape(-1, 1)
            
            # Gradient for weights
            grad_weights = (X.T @ error) / n_samples
            
            # Apply regularization to the gradient
            if self.regularization > 0:
                grad_weights += self.regularization * self.weights / n_samples

            # Update the weights based on gradients
            self.weights -= self.learning_rate * grad_weights

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term
        logits = X @ self.weights
        y_pred = self.sigmoid(logits)
        return (y_pred >= 0.5).astype(int)  # Apply threshold to make prediction

    def score(self, X, y):
        y_pred = self.predict(X).flatten()
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save(self, filepath):
        np.savez(filepath, weights=self.weights)

    def load(self, filepath):
        npzfile = np.load(filepath)
        self.weights = npzfile['weights']
