import numpy as np

class DecisionTree:
    def __init__(self, max_depth=7,min_size=3):
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self,X,y):
        dataset = np.column_stack([X,y])
        self.tree = self.__build_tree(dataset)

    def predict(self,X):
        if len(X.shape) == 1:
            return  self.__predict_one(self.tree,X)
        else:
            y_pred = []
            for x in X:
                y_pred.append(self.__predict_one(self.tree,x))
            return y_pred

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * ' ', node)))

    def __gini_index(self,groups,classes):
        n_instanse = float(sum([ len(group) for group in groups ]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val)/size
                score +=p*p
            gini +=(1.0 - score)*(size/n_instanse)
        return  gini

    def __test_split(self,index, value, dataset):
        left, rigth = [],[]
        for row in dataset:
            if row[index]<value:
                left.append(row)
            else:
                rigth.append(row)
        return left, rigth

    def __get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.__test_split(index,row[index],dataset)
                gini = self.__gini_index(groups,class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return  {'index':b_index,'value':b_value,'groups':b_groups}

    def __to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return  max(set(outcomes), key=outcomes.count)

    def __split(self,node,depth):
        max_depth,min_size, = self.max_depth, self.min_size

        left, right = node['groups']
        del(node['groups'])
        if not left or not  right:
            node['left'] = node['right'] = self.__to_terminal(left+right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.__to_terminal(left), self.__to_terminal(right)
            return

        if len(left) <= min_size:
            node['left'] =self.__to_terminal(left)
        else:
            node['left'] = self.__get_split(left)
            self.__split(node['left'], depth + 1)

        if len(right) <= min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_split(right)
            self.__split(node['right'], depth + 1)

    def __build_tree(self,train):
        root = self.__get_split(train)
        self.__split(root,1)
        return  root


    def __predict_one(self,node,row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__predict_one(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict_one(node['right'], row)
            else:
                return node['right']


if __name__ == '__main__':
    dt = DecisionTree()

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
    tree = dt.fit(dataset[:,:-1],dataset[:,-1])
    #dt.print_tree(tree)
    print(dt.predict(dataset[:,:-1]))