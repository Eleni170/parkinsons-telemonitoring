from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from checks.ChecksSVM import ChecksSVM
from svm.SVMLinearRegressor import SVMLinearRegressor
from svm.SVMNonLinearRegressor import SVMNonLinearRegressor


class ApplicationSVM:

    def __init__(self, data_path, features_start_index):
        data = read_csv(data_path, header=0).values
        number_of_attributes = len(data[0])
        self.x = data[:, features_start_index:number_of_attributes]
        self.t_motor_updrs = data[:, (features_start_index - 2)]
        self.t_total_updrs = data[:, (features_start_index - 1)]

    def main_svm_motor_updrs(self):

        max_iter = ChecksSVM().check_if_valid_max_iter("Set max_iter: ")
        epsilon = ChecksSVM().check_if_valid_epsilon("Set epsilon: ")
        c = ChecksSVM().check_if_valid_c("Set c: ")
        tol = ChecksSVM().check_if_valid_tol("Set tolerance: ")
        kernel = ChecksSVM().check_if_valid_kernel("Set kernel (linear, poly, rbf, sigmoid, precomputed): ")

        if kernel == "linear":
            svm_regressor = SVMLinearRegressor(c, epsilon, tol, max_iter)
        else:

            if kernel == "poly":
                degree = ChecksSVM().check_if_valid_degree("Set degree: ")
            else:
                degree = 3

            if kernel in ["poly", "rbf", "sigmoid"]:
                gamma = ChecksSVM().check_if_valid_gamma("Set gamma: ")
            else:
                gamma = "auto"

            svm_regressor = SVMNonLinearRegressor(c, kernel, gamma, epsilon, tol, degree, max_iter)

        number_of_folds = 9
        mean_squared_error_avg = 0
        mean_absolute_error_avg = 0
        training_time_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_motor_updrs, test_size=0.4)
            training_time_avg = training_time_avg + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
            mean_absolute_error_avg = mean_absolute_error_avg + mean_absolute_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        mean_absolute_error_avg = mean_absolute_error_avg / number_of_folds
        training_time_avg = training_time_avg / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_avg))
        print('Time in average training takes: ' + str(training_time_avg) + ' seconds.')

        self.plot_results_svm('motor_updrs', t_test, predict_test)

    def main_svm_total_updrs(self):

        max_iter = ChecksSVM().check_if_valid_max_iter("Set max_iter: ")
        epsilon = ChecksSVM().check_if_valid_epsilon("Set epsilon: ")
        c = ChecksSVM().check_if_valid_c("Set c: ")
        tol = ChecksSVM().check_if_valid_tol("Set tolerance: ")
        kernel = ChecksSVM().check_if_valid_kernel("Set kernel (linear, poly, rbf, sigmoid, precomputed): ")
        if kernel == "linear":
            svm_regressor = SVMLinearRegressor(c, epsilon, tol, max_iter)
        else:

            if kernel == "poly":
                degree = ChecksSVM().check_if_valid_degree("Set degree: ")
            else:
                degree = 3

            if kernel in ["poly", "rbf", "sigmoid"]:
                gamma = ChecksSVM().check_if_valid_gamma("Set gamma: ")
            else:
                gamma = "auto"

            svm_regressor = SVMNonLinearRegressor(c, kernel, gamma, epsilon, tol, degree, max_iter)

        number_of_folds = 9
        mean_squared_error_avg = 0
        mean_absolute_error_avg = 0
        training_time_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_total_updrs, test_size=0.4)
            training_time_avg = training_time_avg + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
            mean_absolute_error_avg = mean_absolute_error_avg + mean_absolute_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        mean_absolute_error_avg = mean_absolute_error_avg / number_of_folds
        training_time_avg = training_time_avg / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_avg))
        print('Time in average training takes: ' + str(training_time_avg) + ' seconds.')

        self.plot_results_svm('total_updrs', t_test, predict_test)

    def plot_results_svm(self, plt_title_updrs, t_test, predict_test):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Support Vector Regression (' + plt_title_updrs + ')')
        plot0 = plt.plot(t_test, 'r.', label='Test values')
        plot1 = plt.plot(predict_test, label='Predicted values')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Output')
        plt.legend(handles=[plot0[0], plot1[0]])
        plt.show()
