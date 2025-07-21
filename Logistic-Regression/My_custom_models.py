import numpy as np
import matplotlib.pyplot as plt

class My_custom_logisticRegression:

    def __init__(self, lr=0.01, epochs=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.epochs = epochs
        self.losses = []  # to store loss values for visualization

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __binary_cross_entropy(self, y, y_hat):
        eps = 1e-15  # to prevent log(0)
        y_hat = np.clip(y_hat, eps, 1 - eps) 
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.__sigmoid(z)

            # Loss
            loss = self.__binary_cross_entropy(y, y_pred)
            self.losses.append(loss)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.__sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def plot_loss(self):
        plt.plot(range(len(self.losses)), self.losses, label="Loss over Epochs", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy Loss")
        plt.title("Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.show()
        
