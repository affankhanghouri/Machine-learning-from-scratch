import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Multiclass_logisticRegression:

  def __init__(self , X_train , y_train , learning_rate = 0.01 , epochs = 1000):

    self.__weights = None
    self.__bias = None
    self.__X_train = X_train
    self.__y_train = y_train
    self.__learning_rate = learning_rate
    self.__no_of_epochs = epochs
    self.__loss_graph = [] #  for visualizing loss

  @staticmethod
  def __validator(X_train , y_train):

    if len(X_train) != len(y_train):
      raise ValueError('length of X and y should be equal')

    if not isinstance(X_train , (pd.DataFrame , pd.Series , np.ndarray)):
      raise ValueError('invalid datatype for X')

    if not isinstance(y_train,(np.ndarray , pd.DataFrame , pd.Series)):
      raise ValueError('Invalid datatype for y')

    return np.array(X_train) , np.array(y_train)

  def __softmax(self , z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

  def __CrossEntropyLoss(self, y_true, y_pred):
    m = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss

  def fit(self):

    X , y = self.__validator(self.__X_train , self.__y_train)

    # converting y to one hot encoding
    num_classes = len(np.unique(y))
    y_encoded = np.eye(num_classes)[y]  # one-hot encoded

    # initializing weights
    n_samples , n_features = X.shape

    self.__weights = np.zeros((n_features, num_classes))
    self.__bias = np.zeros((1, num_classes))

    # training loop
    for epoch in range(self.__no_of_epochs):

      # forward pass
      logits = np.dot(X, self.__weights) + self.__bias
      y_pred = self.__softmax(logits)

      loss = self.__CrossEntropyLoss(y_encoded, y_pred)
      self.__loss_graph.append(loss)

      # backprop
      dz = (y_pred - y_encoded) / n_samples
      dw = np.dot(X.T, dz)
      db = np.sum(dz, axis=0, keepdims=True)

      # update weights
      self.__weights -= self.__learning_rate * dw
      self.__bias -= self.__learning_rate * db

      if epoch % 100 == 0:
        print(f'[EPOCH : {epoch}  | loss : {loss:.4f}]')

  def predict_probabilty(self, X):
    logits = np.dot(X, self.__weights) + self.__bias
    return self.__softmax(logits)

  def predict(self, X):
    probs = self.predict_probabilty(X)
    return np.argmax(probs, axis=1)
  
  def plot_loss(self):
    if not self.__loss_graph:
        raise ValueError("Model has not been trained yet. No loss to plot.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(self.__loss_graph)), self.__loss_graph, color='blue', linewidth=2)
    plt.title("Cross-Entropy Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()