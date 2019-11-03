from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from checks.Checks import Checks
from knn.KNNRegressor import KNNRegressor


class ApplicationKNN:

    def __init__(self, data_path, features_start_index):
        data = read_csv(data_path, header=0).values
        number_of_attributes = len(data[0])
        self.x = data[:, features_start_index:number_of_attributes]
        self.t_motor_updrs = data[:, (features_start_index - 2)]
        self.t_total_updrs = data[:, (features_start_index - 1)]
        return

    def main_knn_motor_updrs(self):
        k = Checks().check_if_valid_k("Set k: ")
        knn_regressor = KNNRegressor(k)

        number_of_folds = 9
        mean_squared_error_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_motor_updrs, test_size=0.4)
            knn_regressor.fit(x_train, t_train)
            predict_test = knn_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))

        self.plot_results_knn('motor_updrs', t_test, predict_test)

    def main_knn_total_updrs(self):
        k = Checks().check_if_valid_k("Set k: ")
        knn_regressor = KNNRegressor(k)

        number_of_folds = 9
        mean_squared_error_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_total_updrs, test_size=0.4)
            knn_regressor.fit(x_train, t_train)
            predict_test = knn_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))

        self.plot_results_knn('total_updrs', t_test, predict_test)

    def plot_results_knn(self, plt_title_updrs, t_test, predict_test):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Nearest Neighbors Regression (' + plt_title_updrs + ')')
        plot0 = plt.plot(t_test, label='Test values')
        plot1 = plt.plot(predict_test, 'r.', label='Predicted values')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Output')
        plt.legend(handles=[plot0[0], plot1[0]])
        plt.show()