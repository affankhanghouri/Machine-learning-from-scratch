import numpy as np
import pandas as pd

from MyCustomException import InvalidTypeError , InvalidTestSize ,InvalidLength

class My_Label_Encoder:
    """
    Custom class to convert label -> int


    """

    def __init__(self):
        self.__uniqueClasses =None
        self.__label_to_idx =None
        self.__idx_to_label = None

    @staticmethod
    def __validator(y):

        """
        Functon to check whether given input is numpy array / list or not

         """
        
        if isinstance(y , pd.DataFrame):
            y = y.values.ravel()  # flatten if 2D
        if not isinstance(y,(list,np.ndarray,pd.DataFrame)):
            raise InvalidTypeError('only list or numpy array  or pandas dataframe is allowed')
        return list(y)


    @property
    def label_to_idx(self):
        return self.__label_to_idx
    
    @property
    def idx_to_label(self):
        return self.__idx_to_label
    
    def fit(self , y):
        """
        function to fetch unqiue classes in sorted order and than map them into label -> int or int->label

        parameter : (y : list / np.ndarray)
         
        """
        y = self.__validator(y)

        self.__uniqueClasses = sorted(list(set(y))) # sorted all unique classes
        self.__label_to_idx ={element:idx for idx , element in enumerate(self.__uniqueClasses)} #label to idx mapping
        self.__idx_to_label = {idx :label for label , idx in self.__label_to_idx.items()} # idx to label mapping

        return self
    
    def transform(self,y):

        y=self.__validator(y)

        if self.label_to_idx is None:
            raise ValueError('first call fit() and than call tansform()')
        
        try:
            return [self.__label_to_idx[element] for element in y]
        except Exception as e:
            raise ValueError(f'Label : {e.args[0]} was not seen during fit()')
        
    def fit_transform(self,y):

        y = self.__validator(y)

        self.fit(y)
        return self.transform(y)

    def inverse_transform(self , y):

        if self.idx_to_label is None:
            raise ValueError('First call fit() and than call inverse trasform()')

        try:
            return [self.__idx_to_label[idx] for idx in y]
        except Exception as e:
            raise ValueError(f'Index {e.args[0]} was not seen during fit()')
            


class My_custom_train_test_split:

    """
    Custom train_test_split class


    """

     
    def __init__(self,X , y , test_size = 0.2 , random_state = True):
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        self.random_state=random_state
        
        if isinstance(self.test_size , int):
            if not (1  < self.test_size < 100):
                raise InvalidTestSize('Invalid test size')
            
            self.test_size = test_size /100

        if isinstance(self.test_size, float):
            if not (0 < self.test_size < 1):
                raise InvalidTestSize('Invalid test size')

        if len(self.X) != len(self.y):
            raise InvalidLength('Lenght of X and length of y should be equal')

    def split(self):

        test_count = int(self.test_size * len(self.X))  # calculating test_count

        indices = np.arange(len(self.X))  

        if self.random_state: # if user want reproducibilty
            np.random.seed(42)

        np.random.shuffle(indices) # shuffling indices to avoid bias 
        
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]


        X_train = self.X[train_indices]
        y_train = self.y[train_indices]

        X_test = self.X[test_indices]
        y_test = self.y[test_indices]

        return X_train , X_test , y_train , y_test




class my_custom_standard_scaler:


    @staticmethod
    def __validator(X):
        if isinstance(X , pd.DataFrame):
            X = np.array(X,dtype =float)
        elif isinstance(X, np.ndarray):
            X = X.astype(float)
        else:
            raise  InvalidTypeError('Scaling can only be done if input is numpy array or pd.dataframe') 
        
        if X.ndim !=2:
            raise InvalidTypeError('input data should be in 2D form')
        
        return X

    def __init__(self):
        self.mean = None
        self.std = None

    def  fit(self , X):

        X = self.__validator(X)

        self.mean = np.mean(X,axis =0)
        self.std = np.std (X,axis =0)

        return self

    def transform(self , X):
        

        if self.mean is None or self.std is None:
            raise ValueError('first call fit() and than call transform()')
        
        X = self.__validator(X)

        return (X-self.mean) / self.std 

    def fit_transform(self ,X):

        # validating input 
        X= self.__validator(X)


        self.fit(X)
        return self.transform(X)   
 
    def inverse_transform(self , X_scaled):

        X_scaled = self.__validator(X_scaled)

        if self.mean is None or self.std in None:
            raise ValueError('first call fit() and than call transform()')
        
        return (X_scaled * self.std) + self.mean


 
            
  



  


        
    