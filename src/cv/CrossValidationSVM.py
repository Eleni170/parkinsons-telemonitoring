from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from checks.ChecksSVM import ChecksSVM
from svm.SVMLinearRegressor import SVMLinearRegressor
from svm.SVMNonLinearRegressor import SVMNonLinearRegressor


class CrossValidationSVM:

    """
        Replaces template placeholder with values.

        :param data_path: formatted date to display
        :param features_start_index: priority number
        :returns: formatted string
    """
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

    def svm_construction(self):
        max_iter = ChecksSVM().check_if_valid_max_iter("Set max_iter: ")
        epsilon = ChecksSVM().check_if_valid_epsilon("Set epsilon: ")
        c = ChecksSVM().check_if_valid_c("Set c: ")
        tol = ChecksSVM().check_if_valid_tol("Set tolerance: ")
        kernel = ChecksSVM().check_if_valid_kernel("Set kernel (linear, poly, rbf, sigmoid): ")

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

        return svm_regressor

    def cv_svm_motor_updrs(self):

        svm_regressor = self.svm_construction()

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0

        kf = KFold(n_splits=number_of_folds)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_motor_updrs[train_index], self.t_motor_updrs[test_index]

            training_time = training_time + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)

            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")

    def cv_svm_total_updrs(self):

        svm_regressor = self.svm_construction()

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0

        kf = KFold(n_splits=number_of_folds)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_total_updrs[train_index], self.t_total_updrs[test_index]

            training_time = training_time + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)

            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")

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
