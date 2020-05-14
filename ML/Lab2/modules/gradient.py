import numpy as np

class GradientClassification:
    def __init__(self):
        pass

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def net_input(self,x, theta):
        return np.dot(x, theta)

    def probability(self,x, theta):

        return self.sigmoid(self.net_input(x, theta))

    def cost_function(self,x,y,theta):
        m = x.shape[0]
        P  = self.probability(x,theta)
        total_cost = -(1/m)*np.sum(
            y*np.log(P) + (1-y)*np.log(1-P)
        )
        return total_cost

    def gradient(self,x,y,theta):
        m = x.shape[0]
        return (1/m)*np.dot(x.T, self.sigmoid(self.net_input(x,theta)) - y)


    def compute_gradient(self,x,y,theta):
        return self.gradient(x,y,theta)

    def compute_cost_function(self,x,y,theta):
        return  self.cost_function(x,y,theta)

class GradientClassificationRegularizationL2(GradientClassification):
    def __init__(self,C):
        self.lamd = C

    def cost_function(self,x,y,theta):
        m =x.shape[0]
        reg_term = np.sum(theta**2)*self.lamd/(2*m)
        P = self.probability(x,theta)
        total_cost = -(1/m)*np.sum(
            y*np.log(P) + (1-y)*np.log(1-P)
        ) + reg_term
        return total_cost

    def gradient(self,x,y,theta):
        m = x.shape[0]
        grad = (1/m)*np.dot(x.T, self.sigmoid(self.net_input(x,theta)) - y) + theta*self.lamd/m
        return grad


class GradientClassificationRegularizationL1(GradientClassification):
    def __init__(self,C):
        self.lamd = C
    def cost_function(self,x,y,theta):
        m =x.shape[0]
        reg_term = np.abs(theta)*self.lamd
        P = self.probability(x,theta)
        total_cost = -(1/m)*np.sum(
            y*np.log(P) + (1-y)*np.log(1-P)
        ) + reg_term
        return total_cost

    def gradient(self,x,y,theta):
        m = x.shape[0]
        grad = (1/m)*np.dot(x.T, self.sigmoid(self.net_input(x,theta)) - y) + np.sign(theta)*self.lamd/m
        return grad

if __name__ == '__main__':
    import  pandas as pd
    name = '../data/test_data.csv'
    data = pd.read_csv(name)
    data = data.astype(float)

    X = data.iloc[:, :-1].values
    X.dtype='float64'
    y = data.iloc[:, -1].values
    #y = y[:, np.newaxis]
    X = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.array([1,5,1])
    gradien_simple  = GradientClassification()
    #print(f'Simple cost =   {gradien_simple.compute_cost_function(X, y, theta)}')
   # print(f'Simple gradient =   {gradien_simple.compute_gradient(X, y, theta)}')
    print(gradien_simple.sigmoid(100))
   # gradien_reg = GradientClassificationRegularization(lamd=0.5)
    ##print(f'Regular cost =   {gradien_reg.compute_cost_function(X, y, theta)}')
    #print(f'Regular gradient =   {gradien_reg.compute_gradient(X, y, theta)}')
