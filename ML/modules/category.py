class Category:
    #     класс категории для определния вектора y
    def __init__(self, values):
        #         self.category = category
        self.values = values

    def calc_y(self, value):
        #       список заполенненый 1 или 0: 1 если значение в эксперименте равно значанию категории
        y = [1 if value == cat_val else 0 for cat_val in self.values]
        return y


