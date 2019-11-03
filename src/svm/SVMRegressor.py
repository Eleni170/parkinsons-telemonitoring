from sklearn.svm import SVR
import timeit


class SVMRegressor:

    def __init__(self, c, gamma):
        self.C = c
        self.gamma = gamma
        self.svm_regressor = SVR(C=self.C, gamma=self.gamma)

    def fit(self, x_train, t_train):
        start = timeit.default_timer()
        self.svm_regressor.fit(x_train, t_train)
        stop = timeit.default_timer()
        return stop - start

    def predict(self, x_test):
        predict_test = self.svm_regressor.predict(x_test)
        return predict_test
