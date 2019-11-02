from sklearn.svm import SVR


class SVMRegressor:

    def __init__(self, c, gamma):
        self.svm_classifier = None
        self.C = c
        self.gamma = gamma

    def fit(self, x_train, t_train):
        self.svm_classifier = SVR(self.C, gamma=self.gamma)
        self.svm_classifier.fit(x_train, t_train)

    def predict(self, x_test):
        predict_test = self.svm_classifier.predict(x_test)
        return predict_test
