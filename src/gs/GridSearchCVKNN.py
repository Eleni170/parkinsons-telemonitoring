from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


class GridSearchCVKNN:

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

    def gs_knn_motor_updrs(self):
        number_of_folds = 5
        param_grid = {
            'n_neighbors': [1, 2, 3, 4, 5]
        }

        gsc = GridSearchCV(
            estimator=KNeighborsRegressor(),
            param_grid=param_grid,
            cv=number_of_folds
        )

        grid_result = gsc.fit(self.x, self.t_motor_updrs)
        best_params = grid_result.best_params_

        best_svr = KNeighborsRegressor(n_neighbors=best_params["n_neighbors"])

        x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_motor_updrs, test_size=0.4)
        model = best_svr.fit(x_train, t_train)
        predict_test = best_svr.predict(x_test)

        mean_squared_error_value = mean_squared_error(t_test, predict_test)
        mean_absolute_error_value = mean_absolute_error(t_test, predict_test)

        print("Smallest MSE: " + str(mean_squared_error_value))
        print("Smallest MAE: " + str(mean_absolute_error_value))
        print("Best model parameters: " + str(model))

        self.plot_results_knn('motor_updrs', t_test, predict_test)

    def gs_knn_total_updrs(self):
        number_of_folds = 5
        param_grid = {
            'n_neighbors': [1, 2, 3, 4, 5]
        }

        gsc = GridSearchCV(
            estimator=KNeighborsRegressor(),
            param_grid=param_grid,
            cv=number_of_folds
        )

        grid_result = gsc.fit(self.x, self.t_total_updrs)
        best_params = grid_result.best_params_

        best_svr = KNeighborsRegressor(n_neighbors=best_params["n_neighbors"])

        x_train, x_test, t_train, t_test = train_test_split(self.x, self.t_total_updrs, test_size=0.4)
        model = best_svr.fit(x_train, t_train)
        predict_test = best_svr.predict(x_test)

        mean_squared_error_value = mean_squared_error(t_test, predict_test)
        mean_absolute_error_value = mean_absolute_error(t_test, predict_test)

        print("Smallest MSE: " + str(mean_squared_error_value))
        print("Smallest MAE: " + str(mean_absolute_error_value))
        print("Best model parameters: " + str(model))

        self.plot_results_knn('total_updrs', t_test, predict_test)

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
