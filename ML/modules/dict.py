import  numpy as np
import  pandas as pd
from category import  Category

class Dict:
    def __init__(self, dict_=None):
        self.dict = dict_

    def raise_make(self):
        if self.dict == None:
            raise NameError('Словарь не создан')


class CategoriesDict(Dict):

    #     словарь критериев чтобы легко строить y(j)
    def calc_category_y(self, category_key, value):
        self.raise_make()
        return self.dict[category_key].calc_y(value)

    def get_category(self, category_key):
        self.raise_make()
        return self.dict[category_key]

    def make(self, X):
        self.columns = X.columns.values
        categories_dict = {column: Category(X[column].unique()) for column in self.columns}

        self.__make_categories_list(X)

        self.dict = categories_dict
        return self

    @property
    def categories_list(self):
        return self.__categories_list

    @property
    def dictionary(self):
        self.raise_make()
        return self.dict

    def __make_categories_list(self, X):
        self.__categories_list = []

        for column in self.columns:
            self.__categories_list += list(X[column].unique())

        self.__categories_list = np.array(self.__categories_list)
