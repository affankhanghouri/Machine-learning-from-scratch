import numpy as np
from customException import TestSizeError

class My_train_test_split:

    def __init__(self, X, y, test_size=0.2, random_state=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        self.random_state = random_state

        if not isinstance(self.test_size, float):
            raise TestSizeError("Test size should be float like 0.1, 0.2 etc.")

        if not (0 < self.test_size < 1):
            raise TestSizeError("Test size float must be between 0 and 1")
        

    def split(self):
        if len(self.X) != len(self.y):
            raise ValueError("Size of X and y must be equal")

        indices = np.arange(len(self.X))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        np.random.shuffle(indices)

        test_count = int(self.test_size * len(self.X))
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        X_train = self.X[train_indices]
        y_train = self.y[train_indices]

        X_test = self.X[test_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test
