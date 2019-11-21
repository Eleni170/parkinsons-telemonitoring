from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from checks.ChecksSVM import ChecksSVM
from svm.SVMRegressor import SVMRegressor


class CrossValidationSVM:

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
        epsilon = ChecksSVM().check_if_valid_epsilon("Set epsilon: ")
        c = ChecksSVM().check_if_valid_c("Set c: ")
        kernel = ChecksSVM().check_if_valid_kernel("Set kernel (linear, poly, rbf, sigmoid): ")

        if kernel == "poly":
            degree = ChecksSVM().check_if_valid_degree("Set degree: ")
        else:
            degree = 3

        if kernel in ["poly", "rbf", "sigmoid"]:
            gamma = ChecksSVM().check_if_valid_gamma("Set gamma: ")
        else:
            gamma = "auto"

        svm_regressor = SVMRegressor(c, kernel, gamma, epsilon, degree)

        return svm_regressor

    def cv_svm_motor_updrs(self):
        svm_regressor = self.svm_construction()

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0
        number_of_support_vectors = 0

        kf = KFold(n_splits=number_of_folds, shuffle=True)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_motor_updrs[train_index], self.t_motor_updrs[test_index]

            training_time = training_time + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)

            number_of_support_vectors = number_of_support_vectors + len(svm_regressor.svm_regressor.support_vectors_)
            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        number_of_support_vectors = int(number_of_support_vectors / number_of_folds)
        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")
        print("Average number of support vectors: " + str(number_of_support_vectors))

    def cv_svm_total_updrs(self):
        svm_regressor = self.svm_construction()

        number_of_folds = 5
        mean_squared_error_value = 0
        mean_absolute_error_value = 0
        training_time = 0
        number_of_support_vectors = 0

        kf = KFold(n_splits=number_of_folds, shuffle=True)

        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            t_train, t_test = self.t_total_updrs[train_index], self.t_total_updrs[test_index]

            training_time = training_time + svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)

            number_of_support_vectors = number_of_support_vectors + len(svm_regressor.svm_regressor.support_vectors_)
            mean_squared_error_value = mean_squared_error_value + mean_squared_error(t_test, predict_test)
            mean_absolute_error_value = mean_absolute_error_value + mean_absolute_error(t_test, predict_test)

        number_of_support_vectors = int(number_of_support_vectors / number_of_folds)
        mean_squared_error_value = mean_squared_error_value / number_of_folds
        mean_absolute_error_value = mean_absolute_error_value / number_of_folds
        training_time = training_time / number_of_folds

        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_value))
        print("MAE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_absolute_error_value))
        print("Training time takes in average: " + str(training_time) + " seconds.")
        print("Average number of support vectors: " + str(number_of_support_vectors))
