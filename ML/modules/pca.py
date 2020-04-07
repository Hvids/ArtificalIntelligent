from unicodedata import decomposition

import pandas as pd
from sklearn.preprocessing import  StandardScaler
from copy import deepcopy
import numpy as np
from scipy.linalg import fractional_matrix_power
class PCA:
    def __init__(self,n_components = None):
        self.n_components = n_components

    def fit_tranform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self,X):
        scaler = StandardScaler(with_std=False)
        scaler.fit(X)
        X = scaler.transform(X).T
        Z = np.dot(self.__load_matrix_, X).T
        Z = np.array([Z[:,i]/self.__explained_variance_[i] for i in range(self.n_components)])
        return Z.T

    def fit(self,X):
        if self.n_components == None:
            self.n_components = X.shape[1]
        scaler = StandardScaler(with_std=False)
        scaler.fit(X)
        X = scaler.transform(X).T
        cov_matrix = np.cov(X)
        eig_val, eig_vec = np.linalg.eigh(cov_matrix)
        eig_val = eig_val[::-1]
        eig_vec = eig_vec[::-1]
        A = np.diag(eig_val)
        A_sqrt_ = fractional_matrix_power(A,-0.5)
        A_sqrt = fractional_matrix_power(A, 0.5)

        self.__main_component_ = np.dot(np.dot(A_sqrt_,eig_vec.T),X).T
        self.__load_matrix_ = np.dot(eig_vec,A_sqrt)
        k = eig_val.sum()
        self.__explained_variance_ratio_ = np.array([e/k for e in eig_val])
        self.__mean_ = self.__main_component_.mean(axis = 0)
        self.__explained_variance_ = eig_val
        return self


    @property
    def main_component_(self):
        return self.__main_component_

    @property
    def explain_variance_ratio_(self):
        return self.__explained_variance_ratio_[:self.n_components]

    @property
    def load_matrix_(self):
        return self.__load_matrix_

    @property
    def mean_(self):
        return self.__mean_[:self.n_components]

    @property
    def explain_variance_(self):
        return self.__explained_variance_[:self.n_components]

if __name__ == "__main__":
    df = pd.read_csv('pca_test2.csv')
    pca = PCA(n_components=2)
    pca.fit(df)
    t = pca.transform(df)
    print(t)
    print(pca.explain_variance_ratio_)

    from sklearn import decomposition

    pca = decomposition.PCA(n_components=2)
    pca.fit(df)
    print(pca.transform(df))