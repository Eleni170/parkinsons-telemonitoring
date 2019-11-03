from sklearn.svm import SVR


class SVMRegressor:

    def __init__(self, c, gamma):
        self.C = c
        self.gamma = gamma
        self.svm_classifier = SVR(C=self.C, gamma=self.gamma)

    def fit(self, x_train, t_train):
        self.svm_classifier.fit(x_train, t_train)

    def predict(self, x_test):
        predict_test = self.svm_classifier.predict(x_test)
        return predict_test
