import pandas as pd
import numpy as np
import scipy
from scipy.linalg import fractional_matrix_power
from copy import deepcopy
from tags import  Tags
from transformer import  Transformer
from matrix import  BinaryMatrixData
from copy import deepcopy

class ConformityDigitization:
    #     установка меток для категориальных призаков при помощи матрици соответсвий для многомерных

    def transform(self, X):
        transformer = Transformer()
        transformer.transform_categorials(X,self.columns,self.C)
        return  transformer.data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, multidimentionality=False):
        X = deepcopy(X)
        #         X - only caregorial
        self.columns = X.columns.values
        p = X.columns.values.shape[0]
        binary_matrix = BinaryMatrixData()
        binary_matrix = binary_matrix.make(X)
        D = binary_matrix.make_D()

        D_sqrt_ = fractional_matrix_power(D, -0.5)
        D = None
        F = np.dot(((1 / p ** 0.5) * binary_matrix.matrix), D_sqrt_)
        binary_matrix = None
        T_1 = np.dot(F.T, F)
        F = None
        tags = Tags()
        tags.make(T_1,D_sqrt_,multidimentionality)
        T_1 = None
        self.C = tags.tags
        return self



if __name__ == '__main__':
    data = pd.DataFrame({'a': ['a', 'b', 'b'], 'b': [1, 2, 2]})
    binary_matrix = BinaryMatrixData()
    binary_matrix = binary_matrix.make(data)
    con_dig = ConformityDigitization()
    con_dig = con_dig.fit(data[['a','b']])
    data = con_dig.transform(data)
    print(data)
