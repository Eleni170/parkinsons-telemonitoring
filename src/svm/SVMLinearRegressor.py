from sklearn.svm import LinearSVR
import timeit


class SVMLinearRegressor:

    def __init__(self, c, epsilon, max_iter):
        self.C = c
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.svm_regressor = LinearSVR(C=self.C, epsilon=self.epsilon, max_iter=self.max_iter)

    def fit(self, x_train, t_train):
        start = timeit.default_timer()
        self.svm_regressor.fit(x_train, t_train)
        stop = timeit.default_timer()
        return stop - start

    def predict(self, x_test):
        predict_test = self.svm_regressor.predict(x_test)
        return predict_test
