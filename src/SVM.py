from sklearn.svm import SVR


class SVM:

    def __init__(self, C, gamma):
        self.svm_classifier = None
        self.C = C
        self.gamma = gamma

    def classify(self, x_train, t_train):
        self.svm_classifier = SVR(self.C, kernel='rbf', gamma=self.gamma)
        self.svm_classifier.fit(x_train, t_train)

    def predict(self, x_test):
        predict_test = self.svm_classifier.predict(x_test)
        return predict_test
