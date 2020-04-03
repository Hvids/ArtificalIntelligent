import  numpy as np
class MaximizatorScallerProduct:
    def solve(self, T, multidimentionality = False):
        eig, vec = np.linalg.eig(T)
        if multidimentionality:
            C = []
            k = 0
            while True:
                c = 0
                k += 1
                indmax = eig.argmax()
                if eig[indmax] < 1 / p:
                    break
                c = vec[indmax]
                C.append(list(c))

                eig = np.delete(eig, indmax)
                vec = np.delete(vec, indmax, axis=0)

            C = np.array(C).T
            self.decision = C
        else:
            indmax = eig.argmax()
            self.decision = vec[indmax]
        return self
