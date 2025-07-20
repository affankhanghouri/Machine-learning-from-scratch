import numpy as np


class My_custom_LinearRegression:
    """
    Custom Linear Regression class using Gradient Descent.

    - Prediction: y_pred = X * w + b
    - Loss: MSE = mean((y_pred - y_true)^2)
    - Gradients:
        dw = (2/N) * X.T @ (y_pred - y_true)
        db = (2/N) * sum(y_pred - y_true)
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = lr
        self.no_of_epochs = epochs

    def fit(self, X_train, y_train):
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train, dtype=float)

        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train, dtype=float)

        X = X_train
        y = y_train

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)  # corrected to match num_features
        self.bias = 0

        for epoch in range(self.no_of_epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            y = y.ravel()
            y_pred = y_pred.ravel() 

            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print loss occasionally
            if epoch % 100 == 0 or epoch == self.no_of_epochs - 1:
                loss = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X_test):
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test, dtype=float)

        return np.dot(X_test, self.weights) + self.bias
