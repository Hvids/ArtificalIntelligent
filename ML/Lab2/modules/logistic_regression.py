import numpy as np
from gradient import  GradientClassification,GradientClassificationRegularizationL1, GradientClassificationRegularizationL2
from copy import  deepcopy
from scipy.optimize import fmin_tnc
class LogisticRegression:
    l1 = 'l1'
    l2 = 'l2'
    def __init__(self,learning_rate = 0.01, penalty = None, C = 1.0 ):
        self.gradient = self.__set_grad(penalty,C)
        self.learning_rate = learning_rate

    def __set_grad(self,penalty,C):
        if penalty == self.l1:
            return  GradientClassificationRegularizationL1(C=C)
        elif penalty == self.l2:
            return  GradientClassificationRegularizationL2(C=C)
        else:
            return GradientClassification()

    def fit(self, X,y,batch_size = 32):
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(X_.shape[1])
        theta[-1] = 1
        self.parametrs = self.__fit(X_,y,theta,batch_size)

    def __fit(self,X,y,theta,batch_size):
        G = np.zeros(X.shape[1])
        eps = 0.1**10
        alpha = 0.9
        w = deepcopy(theta)
        learning_rate = self.learning_rate
        n_iter = 100000

        #cost_old = self.gradient.compute_cost_function(X,y,w)

        for i in range(n_iter):
            ind = np.random.choice(X.shape[0], batch_size)
            X_ind, y_ind = X[ind, :], y[ind]
            g = self.gradient.compute_gradient(X_ind,y_ind,w)
            G = alpha * G + (1 - alpha) * g ** 2
            w = w - self.learning_rate * g / (np.sqrt(G + eps))
            #cost_new = self.gradient.compute_cost_function(X,y,w)

        return  w



    def predict(self,X):
        if len(X.shape) == 2:
            X_ = np.c_[np.ones((X.shape[0], 1)), X]
        elif len(X.shape)  == 1:
            X_ = np.hstack([1,X])
        else:
            raise  NameError('sizeError')
        theta = self.parametrs

        return  self.gradient.probability(X_,theta)

    def predict_classes(self,X,probab_threshold = 0.5):
        return (self.predict(X) >= probab_threshold).astype(int)


def accuracy(y,y_pred):
    return  np.mean(y == y_pred)
if __name__ == '__main__':
    import pandas as pd

    name = '../data/test_data.csv'
    data = pd.read_csv(name)
    data = data.astype(float)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    lr = LogisticRegression()
    lr.fit(X,y)
    y_pred = lr.predict_classes(X)
    print(accuracy(y_pred,y))