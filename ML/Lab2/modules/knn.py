import kdtree
import copy
from pickle import dump
from pickle import load

import distance as ds

import  numpy as np
import math
from scipy.spatial.distance import cosine

class KNN:
    manhattan = 'manhattan'
    euclidean = 'euclidean'
    chebyshev = 'chebyshev'
    minkowski = 'Minkowski'

    def __init__(self, n_neighbors = 3,dist = 'euclidean', dimensions=3, axis=0, sel_axis=None):
        self.n_neighbors = n_neighbors
        self.dimensions = dimensions
        self.axis = axis
        self.sel_axis = sel_axis
        self.dist = self.__set_dist(dist)

    def fit(self,X,y):
        data = []
        for x in list(X):
            data.append({i:x[i] for i in np.arange(X.shape[1]) })
        self.train_data = data
        self.train_label = list(y)
        self.labels = (set(self.train_label))
        self.class_prb = self._calc_train_class_prb(self.train_label)
        self.kdtree = kdtree.create(
            copy.deepcopy(self.train_data), X.shape[1], self.axis, self.sel_axis)
    def predict(self,X):
        if len(X.shape) == 1:
            return self.__predict_one(X)
        else:
            y_preds = []
            for x in X:
                y_preds.append(self.__predict_one(x))
            return y_preds

    def __predict_one(self,x):
        point = {i:xi for i,xi in enumerate(x)}
        y_pred = self.classify(point=point,k = self.n_neighbors,dist=self.dist)
        return  y_pred
    def __set_dist(self,dist):
        if dist == self.euclidean:
            return ds.EuclideanDistance
        if dist == self.minkowski:
            return ds.MinkowskiDistance
        if dist == self.manhattan:
            return ds.ManhattanDistance
        if dist == self.chebyshev:
            return  ds.ChebyshevDistance
        else:
            return ds.EuclideanDistance

    def _calc_train_class_prb(self, labels_list=None):
        if not labels_list:
            return {}

        n = len(labels_list)
        label_num = len(self.labels)
        prb = {}
        for l in self.labels:
            prb[l] = (labels_list.count(l) + 1.0) / (n + label_num)
        return prb

    def decision(self, neighbors=None):
        if not neighbors:
            return sorted(self.class_prb.items(), key=lambda n: n[1],
                          reverse=True)

        else:
            n = len(neighbors)
            prb = {}
            for label in self.labels:
                prb[label] = 0.0
            for kdnode, dist in neighbors:
                index = self.train_data.index(kdnode.data)
                prb[self.train_label[index]] += 1
            for label in self.labels:
                prb[label] = prb[label] / n
            return sorted(prb.items(), key=lambda n: n[1], reverse=True)

    def classify(self, point=None, k=1, dist=None, prbout=0):
        if not point:
            return []
        neighbors = self.kdtree.search_knn(point, k, dist)
        prb = self.decision(neighbors)
        if prbout == 0:
            return prb[0][0]
        elif prbout == 1:
            return prb

    def visualize_kdtree(self):
        """
        Visualize the kdtree.
        """
        kdtree.visualize(self.kdtree)



if __name__ == '__main__':
    import pandas as pd
    #example_knn()
    name = '../data/test_data.csv'
    data1 = pd.read_csv(name)
    data1 = data1.astype(float)

    X = data1.values
    y = data1.iloc[:, -1].values
    knn = KNN(dist='cosine')
    knn.fit(X,y)
    print(knn.predict(X))