from matrix import  BinaryMatrixData
import  numpy as np
from maximizator_scaller_product import  MaximizatorScallerProduct
from transformer import Transformer
import pandas as pd
from tags import Tags
from scipy.linalg import  fractional_matrix_power
class CorrelationDigitization:

    def transform(self, X):
        transformer = Transformer()
        transformer.transform_categorials(X,self.columns,self.C)
        return  transformer.data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def fit(self,data,multidimentionality=False):
        self.columns = data.columns.values

        p = data.columns.values.shape[0]
        n = data.shape[0]
        binary_matrix = BinaryMatrixData()
        binary_matrix = binary_matrix.make(data)
        D = binary_matrix.make_D()
        Y = binary_matrix.matrix
        D_inv = np.linalg.inv(D)
        D = None
        binary_matrix = None

        R_ = np.dot(Y,D_inv)

        D_inv = None

        YT =Y.T
        Y = None
        R = np.dot(R_,YT)
        YT = None
        R /= n

        R_ = None

        binary_matrix = BinaryMatrixData()
        binary_matrix = binary_matrix.make(data)
        D = binary_matrix.make_D()
        Y = binary_matrix.matrix
        D_inv = np.linalg.inv(D)
        D = None
        binary_matrix = None
        tags = Tags()
        tags.make_with_z(R, D_inv, Y, multidimentionality)
        self.C = tags.tags
        return  self

if __name__ == '__main__':
    data = pd.DataFrame({'a': ['a', 'b', 'b'], 'b': [1, 2, 2]})
    binary_matrix = BinaryMatrixData()
    binary_matrix = binary_matrix.make(data)
    con_dig = CorrelationDigitization()
    con_dig = con_dig.fit(data)
    data = con_dig.transform(data)
    print(data)