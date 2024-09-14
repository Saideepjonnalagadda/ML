import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.loss = []  

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        np.random.seed(42)
        n_rows, n_cols = X.shape
        X = np.hstack([X, np.ones((n_rows, 1))])
        y = y.reshape(-1, y.shape[1] if y.ndim > 1 else 1)
        self.w = np.random.randn(n_cols + 1, y.shape[1])
        lim = int(0.9 * n_rows)
        X_train, X_test = X[:lim], X[lim:]
        y_train, y_test = y[:lim], y[lim:]

        best_weights = self.w
        best_loss = float('inf')
        patience_cnt = 0

        # Training loop
        for epoch in range(max_epochs):
            # Shuffle training data
            ind = np.arange(X_train.shape[0])
            np.random.shuffle(ind)
            X_train_shuffled = X_train[ind]
            y_train_shuffled = y_train[ind]

            #Gradient Descent
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_b = X_train_shuffled[start:end]
                y_b = y_train_shuffled[start:end]
                
                # Make predictions
                y_pred = X_b @ self.weights
                
                # Compute errors
                errors = y_pred - y_b

                # Compute gradients
                grad_weights = X_b.T @ errors / batch_size + regularization * self.w / batch_size
                
                # Update weights
                self.w -= 0.01 * grad_weights

            # Validate
            test_pred = X_test @ self.w
            test_loss = np.mean((test_pred - y_test) ** 2)
            self.loss.append(test_loss)  # Track validation loss

            if test_loss < best_loss:
                best_loss = test_loss
                best_weights = self.w
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break

        self.w = best_weights

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term column
        return X @ self.w

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y.reshape(-1, y.shape[1] if y.ndim > 1 else 1)) ** 2)

    def save(self, filepath):
        np.savez(filepath, weights=self.weights)

    def load(self, filepath):
        npzfile = np.load(filepath)
        self.weights = npzfile['weights']