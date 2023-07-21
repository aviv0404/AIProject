import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multiclass import OneVsOneClassifier

GREATER_THAN = True
LESS_THAN = False
EQUALS = True
NOT_EQUALS = False


class Stump:
    def __init__(self, attributeIndex, value, greater_than=GREATER_THAN, equals=EQUALS):
        self.attributesIndex = attributeIndex
        self.value = value
        self.greater_than = greater_than
        self.equals = equals
        self.n_features_in_ = 1
    def predict(self, x):
        x = x[0]
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
            if costs[best_model_index] >= 0.5:
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
        x = self.x
        if len(x[0]) < model.n_features_in_:
            x = get_poly_x(model, x)
        cost = 0
        for i, row in enumerate(x):
            if model.predict(row.reshape(1, -1)) != self.y[i]:
                cost += w[i]
        return cost

    def predict(self, x):
        if len(self.H) == 0:
            # we're avoiding tryign to create a new error type, 
            # this will raise if adaboost doesn't have a model
            raise UnicodeError 
        y = []
        # we want to set all features in all models to be equal
        # so get all unique models with different number of features:
        Ns = set(model.n_features_in_ for model in self.H)
        dif_models = []
        while len(Ns) > 0:
            for model in self.H:
                if model.n_features_in_ in Ns:
                    dif_models.append(model)
                    Ns.remove(model.n_features_in_)

        X = [get_poly_x(model, x) for model in dif_models]
        for i in range(len(x)):
            avg = 0
            for model, fact in zip(self.H, self.alpha):
                row_reshaped = X[choose_n(model, X)][i, ]
                avg += fact * model.predict(row_reshaped.reshape(1,-1))
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


def get_poly_x(model, x):
    model_features = model.n_features_in_
    x_poly = x 
    i = 1
    while len(x_poly[0]) < model_features:
        x_poly = get_poly_n(i).fit_transform(x)
        i += 1
    return x_poly


def get_poly_n(n):
    return PolynomialFeatures(degree=n, interaction_only=False, include_bias=False)

def choose_n(model, X):
    for i,x in enumerate(X):
        if len(x[0]) == model.n_features_in_:
            return i
    return -1