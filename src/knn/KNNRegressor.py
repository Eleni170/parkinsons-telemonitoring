from sklearn.neighbors import KNeighborsRegressor
import timeit


class KNNRegressor:

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.knn_regressor = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def fit(self, x_train, t_train):
        start = timeit.default_timer()
        self.knn_regressor.fit(x_train, t_train)
        stop = timeit.default_timer()
        return stop - start

    def predict(self, x_test):
        predict_test = self.knn_regressor.predict(x_test)
        return predict_test
