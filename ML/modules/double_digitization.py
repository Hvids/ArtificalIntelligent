from transformer import  Transformer
from matrix import  ConjugateMatrix, ProbabilityMatrix
from scipy.linalg import fractional_matrix_power
import  numpy as np
from tags import  Tags
import pandas as pd
class DoubleDigitization:

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, data):
        transformer = Transformer()
        transformer.transform_categorials(data, self.categorials_name, self.C)
        return transformer.data

    def fit(self, data, first_name, second_name, multidimentionality=False):
        # X,y series
        self.categorials_name = [first_name] + [second_name]
        conjugate_matrix = ConjugateMatrix()
        conjugate_matrix.make(data, first_name, second_name)
        probability_matrix = ProbabilityMatrix()
        probability_matrix.make_form_cobjugate_matrix(conjugate_matrix)
        conjugate_matrix = None

        D1 = probability_matrix.get_D1()
        D2 = probability_matrix.get_D2()
        D1_sqrt_ = fractional_matrix_power(D1, -0.5)
        D2_sqrt_ = fractional_matrix_power(D2, -0.5)
        PH = np.dot(np.dot(D1_sqrt_, probability_matrix.matrix), D2_sqrt_)
        T_1 = np.dot(PH.T, PH)
        T_2 = np.dot(PH,PH.T)
        tags1 = Tags()
        tags2 = Tags()
        tags1.make(T_1, D1_sqrt_, multidimentionality)
        tags2.make(T_2, D2_sqrt_, multidimentionality)
        self.C = np.hstack((tags1.tags,tags2.tags))
        return self

if __name__ == '__main__':
    data = pd.DataFrame({'a': ['a', 'b', 'b'], 'b': [1, 2, 2]})
    con_dig = DoubleDigitization()
    con_dig = con_dig.fit(data,'a','b')
    data = con_dig.transform(data)
    print(data)