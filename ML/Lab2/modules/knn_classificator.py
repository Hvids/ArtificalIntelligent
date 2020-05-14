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
            return self.__predict_one(X_predict)
        y_predicts = []
        for x_predict in X_predict:
            y_predicts.append(self.__predict_one(x_predict))
        return y_predicts




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
    name = '../data/test_data.csv'
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