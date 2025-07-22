import pandas as pd
import numpy as np

class My_custom_label_encoder:

  def __init__(self):
    self.__classes = None
    self.__label_to_idx = None
    self.__idx_to_label = None

  @staticmethod
  def __validator(y):

    if not isinstance(y , (np.ndarray , pd.DataFrame, pd.Series ,list)):
      raise ValueError('Invalid data type ')
    return np.array(y)    

  def fit(self , y):

    y = self.__validator(y)

    # fetching unqiue classes in a sorted order
    self.__classes = sorted(set(y))

    # convert class label -> int
    self.__label_to_idx  = {label:idx for idx , label in enumerate(self.__classes)}

    # converting int -> label
    self.__idx_to_label = {idx:label for label , idx in self.__label_to_idx.items()}

    return self

  def transform(self , y):

    if self.__classes is None:
      raise ValueError('First call fit() and than call transform()')

    y = self.__validator(y)

    try:
      return np.array([self.__label_to_idx[label] for label in y])

    except Exception as e:
      raise ValueError(f'{e.args[0]} label did not appear while fitting.')

  def fit_transform(self , y):
    y = self.__validator(y)
    self.fit(y)
    return self.transform(y)    

  def inverse_transform(self , y_converted):

    if self.__idx_to_label is None:
      raise ValueError('first call fit() go :)')

    y = self.__validator(y_converted)

    try:
      return np.array([self.__idx_to_label[idx ] for idx in y])

    except Exception as e:
      raise ValueError(f'{e.args[0]} invalid index')


# -------------------------------------------------------------------------------





class MismatchSize(Exception):
    pass

class my_train_test_split:

    @staticmethod
    def ConstructorValidator(X, y, test_size):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError(' Invalid X: only numpy array and pandas dataframe allowed.')

        if not isinstance(y, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError(' Invalid y: only numpy array, pandas dataframe, or series allowed.')

        if isinstance(test_size, float):
            if not (0 < test_size < 1):
                raise ValueError(' test_size (float) must be in range (0, 1).')

        if len(X) != len(y):
            raise MismatchSize(' Size Mismatch: Length of X and y must be equal.')

    def __init__(self, X, y, test_size=0.2, random_state=True, shuffle=True):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

        self.ConstructorValidator(X, y, test_size=test_size)

        if isinstance(test_size, int):
            if not (0 < test_size < 100):
                raise ValueError(' test_size (int) must be in range 1 to 99.')
            else:
                self.test_size = self.test_size / 100

        self.X = np.array(X)
        self.y = np.array(y)

    def split(self):
        if self.random_state:
            np.random.seed(42)

        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)

        test_count = int(self.test_size * len(self.X))

        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        X_train = self.X[train_indices]
        y_train = self.y[train_indices]

        X_test = self.X[test_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test

    def __str__(self):
        return (f"my_train_test_split(X_shape={self.X.shape}, y_shape={self.y.shape}, "
                f"test_size={self.test_size}, shuffle={self.shuffle}, random_state={self.random_state})")



#--------------------------------------------------------------------------------------------------





class My_standard_Scalar:

  def __init__(self):
    self.mean = None
    self.std = None

  @staticmethod
  def __validator(X):

    if not isinstance(X, (np.ndarray , pd.DataFrame , pd.Series)):
      raise ValueError('Invalid data type')

    return np.array(X)


  def fit(self , X):
    X = self.__validator(X)

    self.mean = np.mean(X,axis = 0)
    self.std = np.std(X,axis = 0)

    return self


  def transform(self , X):

    if self.mean is None or self.std is None:
      raise ValueError('first call fit() and than call transform()')

    X = self.__validator(X)

    return (X - self.mean ) / self.std # Corrected division by std


  def fit_transform(self,X):

    X = self.__validator(X)

    self.fit(X)
    return self.transform(X)


  def inverse_transform(self ,X_scaled):

    if self.mean is None or self.std is None: # Added check for std
      raise ValueError('first call fit and transform than call inverse_transform()')

    return (X_scaled * self.std) + self.mean