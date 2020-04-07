import numpy as np

class Tags:
    """
        Установка тегов для матрици на основе ее собвственных значений и векторов
    """
    def __init__(self,tags= None):
        self.tags = tags

    def make(self, T, D_sqrt_,multidimentionality = False):
        eig, vec = np.linalg.eig(T)
        vec= vec.T
        if multidimentionality:
            C = []
            k = 0
            while True:
                c = 0
                k += 1
                indmax = eig.argmax()
                if eig[indmax] < 1 / p:
                    break
                c = eig[indmax] ** 0.5 * (np.dot(D_sqrt_, vec[indmax]))
                C.append(list(c))

                eig = np.delete(eig, indmax)
                vec = np.delete(vec, indmax, axis=0)

            C = np.array(C).T
            self.tags = C
        else:
            indmax = eig.argmax()
            self.tags = eig[indmax] ** 0.5 * np.dot(D_sqrt_, vec[indmax])
        return self

    def make_with_z(self, T, D_inv, Y,multidimentionality = False):
        eig, vec = np.linalg.eig(T)
        vec = vec.T
        n = Y.shape[0]
        if multidimentionality:
            C = []
            k = 0
            while True:
                c = 0
                k += 1
                indmax = eig.argmax()
                if eig[indmax] < 1 / p:
                    break
                z = ((n*eig[indmax]) ** 0.5)*vec[indmax]

                c = (1/(eig[indmax] ** 0.5)) * np.dot((np.dot(D_inv, Y.T)),z)
                C.append(list(c))

                eig = np.delete(eig, indmax)
                vec = np.delete(vec, indmax, axis=0)

            C = np.array(C).T
            self.tags = C
        else:
            indmax = eig.argmax()
            z = ((n * eig[indmax]) ** 0.5) * vec[indmax]

            c = (1 / (eig[indmax] ** 0.5)) * np.dot((np.dot(D_inv, Y.T)), z)
            self.tags = c
        return self
