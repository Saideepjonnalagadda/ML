import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000, regularization=0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.regularization = regularization
        self.w = None
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
    def log_loss(self, y_true, y_pred):
        ep = 1e-15
        y_pred = np.clip(y_pred, ep, 1 - ep)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def fit(self, X, y):
        np.random.seed(42)
        n_rows, n_cols = X.shape
        X = np.hstack([X, np.ones((n_rows, 1))])
        self.w = np.random.randn(n_cols + 1, 1)
        for epoch in range(self.max_epochs):
            cal_logit = X @ self.w
            y_pred = self.sigmoid(cal_logit)
            err = y_pred - y.reshape(-1, 1)
            grad_w = (X.T @ err) / n_rows
            if self.regularization > 0:
                grad_w += self.regularization * self.w / n_rows
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        cal_logit = X @ self.w
        y_pred = self.sigmoid(cal_logit)
        return (y_pred >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X).flatten()
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save(self, filepath):
        np.savez(filepath, weights=self.w)

    def load(self, filepath):
        npzfile = np.load(filepath)
        self.w = npzfile['weights']
