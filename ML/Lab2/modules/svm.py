import numpy as np
from copy import  deepcopy
class SVM:
    def __init__(self,num_of_epochs=10000,lr=0.001,C=30):
        self.num_of_epochs = num_of_epochs
        self.lr = lr
        self.C = C

    def fit(self,X,y):
        self.X = np.column_stack((np.ones(len(X)), X))

        self.y = deepcopy(y)

        self.y[self.y==0] = -1

        self.w = np.ones(len(self.X[0]))
        self.__fit()

    def __fit(self):
        for i in range(self.num_of_epochs):
            L, dw = self.get_cost_grads(self.X, self.w, self.y)
            self.w = self.w - self.lr * dw
            
    def distances(self, w, with_lagrange=True):
        distances = self.y * (np.dot(self.X, w)) - 1
        if with_lagrange:
            distances[distances > 0] = 0
        return distances

    def get_cost_grads(self, X, w, y):

        distances = self.distances(w)

        # Get current cost
        L = 1 / 2 * np.dot(w, w) - self.C * np.sum(distances)
        ds = -np.sign(distances)
        p = np.dot(X.T,ds*y)
        dw = np.zeros(len(w))
        dw = w - (self.C*p)
        return L, dw / len(X)

    def predict(self, X):
        X = np.column_stack((np.ones(len(X)), X))
        y = np.sign(X @ self.w)
        y[y==-1] = 0
        return y


if __name__ =='__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = np.genfromtxt('../data/test_svm.csv', dtype=float, delimiter=',')
    np.random.shuffle(data)

    train_y = data[:, 0]
    train_x = data[:, 1:]

#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    svm = SVM(num_of_epochs=10000, lr=1e-3, C=30)
    svm.fit(train_x,train_y)
    p = svm.predict(train_x)
    train_y[train_y == -1] = 0
    p = p - train_y.flatten()

    # Prediction accuracy should be 1.0 for the training set
    print("Accuracy |", len(np.where(p == 0)[0]) / len(p))