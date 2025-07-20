import numpy as np
from customException import TestSizeError , InvalidInputType
import pandas as pd

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


class My_StandardScaler:

    """

    Custom class for scaling data , making custom so in future while working on serious
    projects i can try differnt formaulas , since in sckit learn there is just Z -score foumula using 

    """

    @staticmethod
    def __validate_input(X):
        if isinstance(X , pd.DataFrame):
            X = np.array(X,dtype =float)
        elif isinstance(X, np.ndarray):
            X = X.astype(float)
        else:
            raise  InvalidInputType('Scaling can only be done if input is numpy array or pd.dataframe') 
        
        if X.ndim !=2:
            raise InvalidInputType('input data should be in 2D form')
        
        return X
      




    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self , X):

        """"

        To fetch the mean and std of column features 

        parameters (X can be numpy array or pandas Dataframe)

        """

        X=self.__validate_input(X)

        
        self.mean = X.mean(axis = 0) # axis =0 , bcz scaling applies on each column
        self.std = X.std(axis =0) 

    def transform(self , X):

        """
        to apply the scaling on each column 
        parameter (X can be numpy array or pandas dataframe) 

        """

        if self.mean is  None or self.std is None:
            raise ValueError('first call fit() than call transform()')

        return (X - self.mean) / self.std
    

    def fit_transform(self , X):

        """
        to fit and apply the scaling transformation on each column

        paramter(X can be numpy array or pandas Dataframe)


        """

        X = self.__validate_input(X)
        self.fit(X)
        return self.transform(X)
    






       