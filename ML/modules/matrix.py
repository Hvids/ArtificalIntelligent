import  numpy as np
import  pandas as pd

class Matrix:
    def __init__(self, matrix=None):
        self.matrix = matrix

    def raise_make(self):
        if self.matrix == None:
            raise NameError('Матрица не создана')




class BinaryMatrixData(Matrix):
    # бинарная матрица всех критериев
    def make(self, X):
        for column in X.columns:
            X[column] = X[column].apply(str)
        self.matrix = pd.get_dummies(X).values
        return self

    def make_D(self):
        m = self.matrix.shape[1]
        D = np.zeros(shape=(m, m))
        for i in range(D.shape[1]):
            D[i, i] = self.matrix[:, i].sum()
        return D

class ProbabilityMatrix(Matrix):
    def make(self, X, class_name,categorial_name):
        conjugate_matrix = ConjugateMatrix()
        conjugate_matrix = conjugate_matrix.make(X,categorial_name,categorial_name)
        self.matrix = conjugate_matrix.matrix/conjugate_matrix.n
        self.m1 = conjugate_matrix.m1
        self.m2 = conjugate_matrix.m1
        return  self
    def make_form_cobjugate_matrix(self,conjugate_matrix):
        self.matrix =  conjugate_matrix.matrix / conjugate_matrix.n
        self.m1 = conjugate_matrix.m1
        self.m2 = conjugate_matrix.m1
        return self

    def get_D1(self):
        # self.raise_make()
        f_row = self.matrix.sum(axis=1)
        D1 = np.diag(f_row)
        return  D1

    def get_D2(self):
        # self.raise_make()
        D1 = np.zeros(shape=(self.m1, self.m2))
        f_row = self.matrix.sum(axis=0)
        D1 = np.diag(f_row)
        return D1



class ConjugateMatrix(Matrix):


    def make(self,data,class_name,categorial_name):
        Y = data[class_name].unique()
        X = data[categorial_name].unique()
        m1, m2 = X.shape[0], Y.shape[0]
        M = np.zeros(shape=(m2, m1))
        for j in range(m2):
            for i in range(m1):
                indx = (data[categorial_name] == X[i]) & (data[class_name] == Y[j])
                M[j][i] = data[indx][categorial_name].count()
        n = M.sum()
        self.matrix = M
        self.n = n
        self.m1 = m1
        self.m2 = m2
        return self

    def __truediv__(self, other):
        self.matrix /= other
        return self

    def get_D1(self):
        # self.raise_make()
        D1 = np.zeros(shape=(self.m1,self.m2))
        f_row = self.matrix.sum(axis=1)/self.n
        D1 = np.diag(f_row)
        return  D1

    def get_D2(self):
        # self.raise_make()
        D1 = np.zeros(shape=(self.m1, self.m2))
        f_row = self.matrix.sum(axis=0) / self.n
        D1 = np.diag(f_row)
        return D1

