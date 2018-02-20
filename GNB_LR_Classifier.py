import numpy as np
import copy


class LogisticRegressionClassifier:
    def __init__(self,lr=0.002,iterations=10):
        self._pos_rows = None
        self._neg_rows = None
        self._pos_pobability = 0
        self._neg_pobability = 0
        self._parameters = 0
        self._samples = 0
        self._lr = lr
        self._iterations = iterations
        self._weights = None
        self._X = None
        self.lastPreds = None


    def _logistic_regressor(self,params):
        return 1-(1/(1 + np.exp(np.inner(self._weights,params))))

    def fit(self,X,y):
        self._parameters = X.shape[1]
        self._samples = X.shape[0]
        self._X = np.insert(X,0,1,axis=1)
        self._weights = np.zeros(self._parameters+1)
        for i in range(self._iterations):
            pred_y = np.apply_along_axis(self._logistic_regressor,1,self._X)
            error = y-pred_y
            self._weights += self._lr*np.sum(self._X.T*error, axis=1)

    def predict(self,X):
        dat = np.insert(X,0,1,axis=1)
        self.lastPreds = np.round(np.apply_along_axis(self._logistic_regressor, 1, dat),0)
        return self.lastPreds


    def accuracy(self,y):
        return (y.shape[0] - np.sum(np.abs(y - self.lastPreds)))/y.shape[0]



class NaiveBayesClassifier:
    def __init__(self):
        self._pos_rows = None
        self._neg_rows = None
        self._pos_pobability = 0
        self._neg_pobability = 0
        self._pos_rows_mean = None
        self._neg_rows_mean = None
        self._pos_rows_var = None
        self._neg_rows_var = None
        self._parameters = 0
        self._samples = 0
        self.lastPreds = None

    def __get_conditional_parameter(self, condition, X, y):
        return X[y == condition]

    def __gaussian_probability_func(self, means, vars, params):
        return (1 / (np.sqrt(vars * 2 * np.pi))) * np.exp(- ((params - means) ** 2 / (2 * vars)))

    def __naive_bayesian_probability(self,params):
        denominator = self._pos_pobability * np.prod(self.__gaussian_probability_func(self._pos_rows_mean, self._pos_rows_var, params)) \
                      + self._neg_pobability * np.prod(self.__gaussian_probability_func(self._neg_rows_mean, self._neg_rows_var, params))
        numerator = self._pos_pobability * np.prod(self.__gaussian_probability_func(self._pos_rows_mean, self._pos_rows_var, params))
        return numerator / denominator

    def fit(self, X,y):
        self._parameters = X.shape[1]
        self._samples = X.shape[0]
        self._pos_rows = self.__get_conditional_parameter(1,X,y)
        self._neg_rows = self.__get_conditional_parameter(0,X,y)
        self._pos_pobability = self._pos_rows.shape[0] / self._samples
        self._neg_pobability = self._neg_rows.shape[0] / self._samples
        self._pos_rows_mean = self._pos_rows.mean(axis=0)
        self._neg_rows_mean = self._neg_rows.mean(axis=0)
        self._pos_rows_var = self._pos_rows.var(axis=0)
        self._neg_rows_var = self._neg_rows.var(axis=0)


    def predict(self,X):
        self.lastPreds = np.round(np.apply_along_axis(self.__naive_bayesian_probability,1,X),0)
        return self.lastPreds

    def accuracy(self,y):
        return (y.shape[0] - np.sum(np.abs(y - self.lastPreds)))/y.shape[0]


def ThreeFoldCrossValidation(data,models):
    model_accuracies = []
    shuffled = copy.deepcopy(data)
    np.random.shuffle(shuffled)
    sample_size = int(shuffled.shape[0]/3)
    for i in range(3):
        test_idx = range(i*sample_size, (i+1)*sample_size)
        test = shuffled[test_idx]
        train = np.delete(shuffled,test_idx,axis=0)
        accuracies = []
        for m in range(len(models)):
            models[m].fit(train[:,0:-1],train[:,-1])
            models[m].predict(test[:,0:-1])
            accuracies.append(models[m].accuracy(test[:,-1]))
        model_accuracies.append(accuracies)
    return models,model_accuracies

