import numpy as np
from decision_tree import  DecisionTree
from collections import  Counter
import math
class RandomForest:
    def __init__(self,n_estimators=10,samples = None, max_depth=7,min_size=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_size = min_size
        self.samples = samples
    def fit(self,X,y):
        self.decision_trees = []

        for _ in range(self.n_estimators):

            ind_row = np.random.choice(X.shape[0],X.shape[0]) if self.samples == None else np.random.choice(X.shape[0],self.samles)
            ind_col = np.random.choice(X.shape[1],int(math.sqrt(X.shape[1])),replace = False)
            X_est = X[ind_row]
            X_est = X_est[:,ind_col]

            y_est = y[ind_row]
            dt = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            dt.fit(X_est,y_est)
            self.decision_trees.append(dt)
    def predict(self,X):
        result = None
        if len(X.shape) == 1:
            result = self.__predict_one(X)
        else:
            y_preds = []
            for x in X:
                y_pred = self.__predict_one(x)
                y_preds.append(y_pred)
            result = y_preds
        return  result

    def __predict_one(self,x):
        y_preds = []
        for dt in self.decision_trees:
            y_pred = dt.predict(x)
            y_preds.append(y_pred)
        voices = Counter(y_preds)
        y_result = voices.most_common()[0][0]
        return  y_result

if __name__ == '__main__':
    dataset = np.array([[2.771244718, 1.784783929, 0],
                        [1.728571309, 1.169761413, 0],
                        [3.678319846, 2.81281357, 0],
                        [3.961043357, 2.61995032, 0],
                        [2.999208922, 2.209014212, 0],
                        [7.497545867, 3.162953546, 1],
                        [9.00220326, 3.339047188, 1],
                        [7.444542326, 0.476683375, 1],
                        [10.12493903, 3.234550982, 1],
                        [6.642287351, 3.319983761, 1]])
    X = dataset[:,:-1]
    y = dataset[:,-1]
    rf = RandomForest(n_estimators=10000)
    rf.fit(X,y)
    print(rf.predict(X))