from scipy.spatial.distance import cosine
import pandas as pd
import  numpy as np
from collections import Counter

def eulera(v,u):
    return np.linalg.norm(v-u)


class KNNClassificator:
    cosine = 'cosine'
    eulera = 'eulera'
    def __init__(self,n_neighbors = 5 ,metrics='cosine'):
        self.metrics = self.__set_metrics(metrics)
        self.n_neighbors = n_neighbors

    def fit(self,X,y):
        self.X = X
        self.y = y
        self.x_index = np.arange(0, X.shape[0])
    def predict(self,X_predict):
        if len(X_predict.shape)==1:
            return self.__pridict_one(X_predict)
        y_predicts = []
        for x_preict in X_predict:
            return self.__predict_many(X_predict)
        return y_predicts

    def __predict_many(self,X_predicts):
        x_fit_indexes = self.x_index
        x_predict_indexes = np.arange(-X_predicts.shape[0],0,1)
        indexes = list(x_fit_indexes) + list(x_predict_indexes)

        uniun_fits_predicts = list(self.X) + list(X_predicts)
        ones = np.ones(self.X.shape[1])
        sizes = list(map( lambda x: self.metrics(x,ones),uniun_fits_predicts))

        tuple_ind_size = list(zip(indexes, sizes))
        tuple_ind_size_sorted = sorted(tuple_ind_size,key=lambda x:x[1])

        indexes = [i for i,_ in tuple_ind_size_sorted]
        sizes = [i for _,i in tuple_ind_size_sorted]

        y_preds = []

        for x_predict_index in x_predict_indexes:
            ind_pred = indexes.index(x_predict_index)
            voice_n_neigh = []
            n = 0
            for ind_fit in indexes[self.n_neighbors//2-ind_pred:]:
                if ind_fit < 0:
                    continue
                n+=1
                if n>self.n_neighbors:
                    break
                voice_n_neigh.append(self.y[ind_fit])
            y_predict = Counter(voice_n_neigh).most_common()[0][0]
            y_preds.append(y_predict)
        return  y_preds




    def __set_metrics(self,metrics):
        if metrics == self.cosine:
            return cosine
        else:
            return eulera

    def __predict_one(self,x_predict):
        sizes = np.array(list(
            map(lambda x:self.metrics(x,x_predict),self.X)
        ))
        x_index_sizes = tuple(zip(self.x_index,sizes))
        n_neighbors_x_index_size = sorted(x_index_sizes, key=lambda x_index_size:x_index_size[1])[:self.n_neighbors]
        knn_index = [i for i,_ in n_neighbors_x_index_size]
        y_n_neigbors = self.y[knn_index]
        y_predict = Counter(y_n_neigbors).most_common()[0][0]
        return y_predict




if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import  accuracy_score
    name = '../test_data.csv'
    data = pd.read_csv(name)
    data = data.astype(float)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    knn = KNNClassificator()
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print(f"accuracy = {acc}")
    knn