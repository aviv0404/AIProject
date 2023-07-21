import numpy as np

GREATER_THAN = True
LESS_THAN = False
EQUALS = True
NOT_EQUALS = False


class Stump:
    attributesIndex = -1
    value = -1
    greater_than = GREATER_THAN
    equals = EQUALS

    def __init__(self, attributeIndex, value, greater_than=GREATER_THAN, equals=EQUALS):
        self.attributesIndex = attributeIndex
        self.value = value
        self.greater_than = greater_than
        self.equals = equals

    def predict(self, x):
        if self.equals:
            if self.greater_than:
                return 1 if x[self.attributesIndex] >= self.value else -1
            else:
                return 1 if x[self.attributesIndex] <= self.value else -1
        else:
            if self.greater_than:
                return 1 if x[self.attributesIndex] > self.value else -1
            else:
                return 1 if x[self.attributesIndex] < self.value else -1
            
    def to_string(self):
        return 'attr ind:{}\tval: {}\tgreater than?{}'.format(self.attributesIndex, self.value, self.greater_than)


class Adaboost:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = len(x)
        self.n = len(x[0])
        self.H = []
        self.alpha = []

    def fit(self, max_iter=1000, min_cost_value=10 ** (-4), models=[]):
        w = np.full(self.m, 1 / self.m)
        for i in range(len(self.x[0])):
            for val in set(self.x[i, ]):
                models.append(
                    Stump(i, value=val, greater_than=GREATER_THAN, equals=EQUALS)
                )
                models.append(
                    Stump(i, value=val, greater_than=LESS_THAN, equals=NOT_EQUALS)
                )
        
        i = 0
        error_score = []
        while self.stop_condition(i, max_iter, min_cost_value, self.cost_function(w)):
            costs = []
            for model in models:
                costs.append(self.loss_function_model(model, w))
            best_model_index = np.argmin(costs)
            if costs[best_model_index] <= 0.5:
                break
            self.H.append(models[best_model_index])
            self.alpha.append(
                0.5 * np.log((1 - costs[best_model_index]) / costs[best_model_index]))
            exact = self.y == self.predict(self.x)
            mistake = self.y != self.predict(self.x)
            error_score.append(self.cost_function(w))
            w[exact] *= ((2*(1 - error_score[-1]))**(-1))
            w[mistake] *= ((2*(error_score[-1]))**(-1))            
                    
            i += 1

    def stop_condition(self, i, max_iter, min_cost_value, current_cost_value):
        return True if max_iter > i and current_cost_value > min_cost_value else False

    def loss_function_model(self, model, w):
        cost = 0
        for i, row in enumerate(self.x):
            if model.predict(row.reshape(1, -1)) != self.y[i]:
                cost += w[i]
        return cost

    def predict(self, x):
        if len(self.H) == 0:
            raise RuntimeError
        y = []
        for row in x:
            avg = 0
            for model, fact in zip(self.H, self.alpha):
                avg += fact * model.predict(row)
            y.append(1 if avg > 0 else -1)
        return np.array(y)

    def cost_function(self, w):
        if len(self.H) == 0:
            return 1000
        cost = 0
        for weight, real, predicted in zip(w, self.y, self.predict(self.x)):
            if real != predicted:
                cost += weight
        return cost
