from dict import  CategoriesDict
from copy import deepcopy
class Transformer:
    """
    Преобразователь данных
    """
    def transform_categorials(self,data,columns, tags):
        X = deepcopy(data)
        categories_dict = CategoriesDict()
        categories_dict = categories_dict.make(X[columns])
        transform_dict = dict(zip(categories_dict.categories_list, tags))
        return self.__make(X,columns,transform_dict)

    def transform_categorial(self,data,categorial_name, tags):
        X = deepcopy(data)
        categories_list = X[categorial_name].unique()
        categories_list  = [str(x) for x in categories_list]
        transform_dict= dict(zip(categories_list, tags))
        return self.__make(X,[categorial_name],transform_dict)

    def __make(self,X,columns,transform_dict):
        for column in columns:
            t = X[column].apply(lambda x: transform_dict[str(x)].real)
            X[column] = t
        self.__data = X
        return self

    @property

    def data(self):
        return deepcopy(self.__data)