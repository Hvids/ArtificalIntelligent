import pandas as pd
from scipy.linalg import  fractional_matrix_power
import  numpy as np
from matrix import  ConjugateMatrix, ProbabilityMatrix
from tags import  Tags
from transformer import  Transformer
class CentrifugalDigitization:
    """
     Установка метки для категориального признака на основе второго(класса)
    """

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self,data):
        transformer = Transformer()
        transformer.transform_categorial(data,self.categorial_name,self.C)
        return transformer.data
    def fit(self,data, class_name,categorial_name,multidimentionality = False):
# X,y series
        self.categorial_name = categorial_name
        conjugate_matrix = ConjugateMatrix()
        conjugate_matrix.make(data, class_name,categorial_name)
        probability_matrix = ProbabilityMatrix()
        probability_matrix.make_form_cobjugate_matrix(conjugate_matrix)
        conjugate_matrix.get_D1()
        # conjugate_matrix = None

        D1 = probability_matrix.get_D1()
        D2 = probability_matrix.get_D2()
        D1_sqrt_ = fractional_matrix_power(D1,-0.5)
        D2_sqrt_ = fractional_matrix_power(D2,-0.5)
        PH = np.dot(np.dot(D1_sqrt_, probability_matrix.matrix), D2_sqrt_)
        T_1 = np.dot(PH.T,PH)
        # T_2 = np.dot(PH,PH.T)
        tags = Tags()
        tags.make(T_1,D2_sqrt_,multidimentionality)
        self.C = tags.tags
        return self


if __name__ =="__main__":
    data = pd.DataFrame({'a': ['a', 'b', 'a'], 'b': [1, 2, 3,]})
    print(data)
    cen_dig = CentrifugalDigitization()
    cen_dig.fit(data,'b','a')
    print(data)
    print(cen_dig.transform(data))
