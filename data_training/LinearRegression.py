from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression:

    def __init__(self, input_x_train, input_y_train, input_x_test, input_y_test):

        self.input_x = input_x_train
        self.input_y = input_y_train

        self.input_x_test = input_x_test
        self.input_y_test = input_y_test

    def train_normal_linear_regression(self):

        reg = linear_model.LinearRegression()
        reg.fit(self.input_x, self.input_y)

        y_pred = reg.predict(self.input_x_test)

        print('mean_square_error: %s \n' % mean_squared_error(self.input_y_test, y_pred))
        print('r2 score: %.2f \n' % r2_score(self.input_y_test, y_pred))
