
import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        np.random.seed(42)
        
        # Initialize variables
        n_samples, n_features = X.shape
        X = np.hstack([X, np.ones((n_samples, 1))])  # Add 1 for bias term
        y = y.reshape(-1, 1)  # Ensuring y is a column vector
        self.weights = np.random.randn(n_features + 1, 1)

        # Split data into training and validation set (90% train, 10% validation)
        split_at = int(0.9 * n_samples)
        X_train, X_val = X[:split_at], X[split_at:]
        y_train, y_val = y[:split_at], y[split_at:]

        best_weights = self.weights
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(max_epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch gradient descent
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                # Make predictions
                y_pred = X_batch @ self.weights
                
                # Compute errors
                errors = y_pred - y_batch
                
                # Compute gradients
                grad_weights = X_batch.T @ errors / batch_size + regularization * self.weights / batch_size
                
                # Update weights
                self.weights -= 0.01 * grad_weights

            # Validate
            val_pred = X_val @ self.weights
            val_loss = np.mean((val_pred - y_val) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break

        self.weights = best_weights

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term column
        return X @ self.weights

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y.reshape(-1, 1)) ** 2)

    def save(self, filepath):
        np.savez(filepath, weights=self.weights)

    def load(self, filepath):
        npzfile = np.load(filepath)
        self.weights = npzfile['weights']
