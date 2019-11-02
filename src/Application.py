from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.Checks import Checks
from src.SVMRegressor import SVMRegressor

import matplotlib.pyplot as plt


class Application:

    def __init__(self, data_path, features_start_index):
        data = read_csv(data_path, header=0).values
        number_of_attributes = len(data[0])
        self.x = data[:, features_start_index:number_of_attributes]
        self.t_motor_updrs = data[:, (features_start_index - 2)]
        self.t_total_updrs = data[:, (features_start_index - 1)]
        return

    def main(self, updrs_index):
        t_updrs_dictionary = {0: self.t_motor_updrs, 1: self.t_total_updrs}
        t_updrs = t_updrs_dictionary.get(updrs_index)

        c = Checks().check_if_valid_c("Set c: ")
        gamma = Checks().check_if_valid_gamma("Set gamma: ")
        svm_regressor = SVMRegressor(c, gamma)

        number_of_folds = 9
        mean_squared_error_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(self.x, t_updrs, test_size=0.4)
            svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))

        plt_title_updrs_dictionary = {0: "motor_updrs", 1: "total_updrs"}
        plt_title_updrs = plt_title_updrs_dictionary.get(updrs_index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Support Vector Regression (' + plt_title_updrs + ')')
        plot0 = plt.plot(t_test, label='Test values')
        plot1 = plt.plot(predict_test, 'r.', label='Predicted values')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Output')
        plt.legend(handles=[plot0[0], plot1[0]])
        plt.show()


if __name__ == '__main__':
    application = Application('../dataset/parkinsons_updrs.data', 6)
    application.main(0)
    application.main(1)
