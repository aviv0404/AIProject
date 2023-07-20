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


class Adaboost:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = len(x)
        self.n = len(x[0])
        self.H = []
        self.alpha = []

    def fit(self, max_iter=1000, min_cost_value=10 ** (-4)):
        w = np.full(self.m, 1 / self.m)
        models = []
        for i in range(len(self.x[0])):
            for val in set(self.x[i,]):
                models.append(
                    Stump(i, value=val, greater_than=GREATER_THAN, equals=EQUALS)
                )
                models.append(
                    Stump(i, value=val, greater_than=GREATER_THAN, equals=EQUALS)
                )
        i, current_cost_value, acc = 0, 100, []
        while self.stop_condition(i, max_iter, min_cost_value, current_cost_value):
            error_score = self.m
            costs = []
            for model in models:
                costs.append(self.loss_function_stump(model, w))
            best_model_index = np.argmin(costs)
            self.H.append(models[best_model_index])
            self.alpha.append(
                0.5 * np.log((1 - costs[best_model_index]) / costs[best_model_index])
            )
            exact = self.y == self.predict(self.x)
            mistake = self.y != self.predict(self.x)
            w[exact] = 0.5 * w[exact] * (1 - self.cost_function(w[exact]))
            w[mistake] = 0.5 * w[mistake] * (self.cost_function(w[mistake]))
            i += 1

    def stop_condition(self, i, max_iter, min_cost_value, current_cost_value):
        return True if max_iter > i and current_cost_value > min_cost_value else False

    def loss_function_stump(self, stump, w):
        cost = 0
        for i, row in enumerate(self.x):
            if stump.predict(row) != self.y[i]:
                cost += w[i]
        return np.array(cost)

    def predict(self, x):
        y = []
        for row in x:
            avg = 0
            for model, fact in zip(self.H, self.alpha):
                avg += fact * model.predict(row)
            y.append(1 if avg > 0 else -1)
        return np.array(y)

    def cost_function(self, w):
        cost = 0
        for weight, real, predicted in zip(w, self.y, self.predict(self.x)):
            cost += weight if real == predicted else 0
        return cost