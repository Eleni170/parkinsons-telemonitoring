from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.Checks import Checks
from src.SVMRegressor import SVMRegressor

import matplotlib.pyplot as plt


class Application:

    def __init__(self):
        return

    @staticmethod
    def main():
        data = read_csv('../dataset/parkinsons_updrs.data', header=0).values
        number_of_attributes = len(data[0])
        number_of_patterns = len(data)
        x = data[:, 6:number_of_attributes]
        t_motor_updrs = data[:, 4]

        c = Checks().check_if_valid_c("Set c: ")
        gamma = Checks().check_if_valid_gamma("Set gamma: ")
        svm_regressor = SVMRegressor(c, gamma)

        number_of_folds = 9
        mean_squared_error_avg = 0
        for _ in range(number_of_folds):
            x_train, x_test, t_train, t_test = train_test_split(x, t_motor_updrs, test_size=0.4)
            svm_regressor.fit(x_train, t_train)
            predict_test = svm_regressor.predict(x_test)
            mean_squared_error_avg = mean_squared_error_avg + mean_squared_error(t_test, predict_test)
        mean_squared_error_avg = mean_squared_error_avg / number_of_folds
        print("MSE of cross validation with " + str(number_of_folds) + " folds is " + str(mean_squared_error_avg))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Support Vector Regression')
        plot0 = plt.plot(t_test, label='Test values')
        plot1 = plt.plot(predict_test, 'r.', label='Predicted values')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Output')
        plt.legend(handles=[plot0[0], plot1[0]])
        plt.show()


if __name__ == '__main__':
    Application().main()
