from pandas import read_csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class GridSearchCVSVM:

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

    def gs_svm_motor_updrs(self):

        number_of_folds = 5
        param_grid = {
            'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32],
            'gamma': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
        }

        gsc = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid,
            cv=number_of_folds
        )

        grid_result = gsc.fit(self.x, self.t_motor_updrs)
        best_params = grid_result.best_params_

        best_svr = SVR(C=best_params["C"], gamma=best_params["gamma"])

        x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_motor_updrs, test_size=0.4)
        model = best_svr.fit(x_train, t_train)
        predict_test = best_svr.predict(x_test)
        number_of_support_vectors = len(best_svr.support_vectors_)

        mean_squared_error_value = mean_squared_error(t_test, predict_test)
        mean_absolute_error_value = mean_absolute_error(t_test, predict_test)

        print("Smallest MSE: " + str(mean_squared_error_value))
        print("Smallest MAE: " + str(mean_absolute_error_value))
        print("Number of Support Vectors: "+str(number_of_support_vectors))
        print("Best model parameters: " + str(model))

        self.plot_results_svm('motor_updrs', t_test, predict_test)

    def gs_svm_total_updrs(self):

        number_of_folds = 5
        param_grid = {
            'C': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32],
            'gamma': [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
        }

        gsc = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid,
            cv=number_of_folds
        )

        grid_result = gsc.fit(self.x, self.t_total_updrs)
        best_params = grid_result.best_params_

        best_svr = SVR(C=best_params["C"], gamma=best_params["gamma"])

        x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_total_updrs, test_size=0.4)
        model = best_svr.fit(x_train, t_train)
        predict_test = best_svr.predict(x_test)
        number_of_support_vectors = len(best_svr.support_vectors_)

        mean_squared_error_value = mean_squared_error(t_test, predict_test)
        mean_absolute_error_value = mean_absolute_error(t_test, predict_test)

        print("Smallest MSE: " + str(mean_squared_error_value))
        print("Smallest MAE: " + str(mean_absolute_error_value))
        print("Number of Support Vectors: " + str(number_of_support_vectors))
        print("Best model parameters: " + str(model))

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
