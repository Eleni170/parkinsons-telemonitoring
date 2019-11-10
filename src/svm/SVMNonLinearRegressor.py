from sklearn.svm import SVR
import timeit


class SVMNonLinearRegressor:

    def __init__(self, c, kernel, gamma, epsilon, tol, degree, max_iter):
        self.C = c
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.tol = tol
        self.degree = degree
        self.max_iter = max_iter
        self.svm_regressor = SVR(C=self.C, kernel=self.kernel, gamma=self.gamma, epsilon=epsilon, tol=self.tol,
                                 degree=self.degree, max_iter=self.max_iter)

    def fit(self, x_train, t_train):
        start = timeit.default_timer()
        self.svm_regressor.fit(x_train, t_train)
        stop = timeit.default_timer()
        return stop - start

    def predict(self, x_test):
        predict_test = self.svm_regressor.predict(x_test)
        return predict_test
