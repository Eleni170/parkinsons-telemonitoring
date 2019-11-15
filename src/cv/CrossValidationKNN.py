from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from checks.ChecksKNN import ChecksKNN
from knn.KNNRegressor import KNNRegressor


class CrossValidationKNN:

    def __init__(self, data_path, features_start_index):
        data = read_csv(data_path, header=0).values
        number_of_attributes = len(data[0])
        self.x = data[:, features_start_index:number_of_attributes]
        self.t_motor_updrs = data[:, (features_start_index - 2)]
        self.t_total_updrs = data[:, (features_start_index - 1)]

        self.x = StandardScaler().fit_transform(self.x)
        self.t_motor_updrs = StandardScaler().fit_transform(self.t_motor_updrs.reshape(-1, 1))
        self.t_motor_updrs = self.t_motor_updrs.reshape(-1, )
        self.t_total_updrs = StandardScaler().fit_transform(self.t_total_updrs.reshape(-1, 1))
        self.t_total_updrs = self.t_total_updrs.reshape(-1, )

    def cv_knn_motor_updrs(self):
        k = ChecksKNN().check_if_valid_k("Set k: ")
        knn_regressor = KNNRegressor(k)

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0

        kf = KFold(n_splits=number_of_folds)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_motor_updrs[train_index], self.t_motor_updrs[test_index]

            training_time = training_time + knn_regressor.fit(x_train, t_train)
            predict_test = knn_regressor.predict(x_test)

            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")

    def cv_knn_total_updrs(self):
        k = ChecksKNN().check_if_valid_k("Set k: ")
        knn_regressor = KNNRegressor(k)

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0

        kf = KFold(n_splits=number_of_folds)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_total_updrs[train_index], self.t_total_updrs[test_index]

            training_time = training_time + knn_regressor.fit(x_train, t_train)
            predict_test = knn_regressor.predict(x_test)

            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")

    def plot_results_knn(self, plt_title_updrs, t_test, predict_test):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Nearest Neighbors Regression (' + plt_title_updrs + ')')
        plot0 = plt.plot(t_test, 'r.', label='Test values')
        plot1 = plt.plot(predict_test, label='Predicted values')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Output')
        plt.legend(handles=[plot0[0], plot1[0]])
        plt.show()
