from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class RandomForestRegression:

    def __init__(self, x_train, y_train, x_test, y_test, max_depth=30):
        self.regr_random_forest = \
            RandomForestRegressor(max_depth=max_depth, random_state=2, verbose=2, n_estimators=100)
        self.regr_random_forest.fit(x_train, y_train)
        self.x_test = x_test
        self.y_test = y_test

    def start_train(self):
        y_pred = self.regr_random_forest.predict(self.x_test)
        print("R2 score: %.2f" % r2_score(self.y_test, y_pred))
