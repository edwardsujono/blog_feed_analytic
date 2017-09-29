from preprocessing.DataPreprocess import DataPreprocess
from data_training.LinearRegression import LinearRegression
from data_training.NeuralNetwork import NeuralNetwork
from data_training.RandomForestRegression import RandomForestRegression
from data_visualization.data_visualize import DataVisualization

if __name__ == "__main__":

    data_preprocess = DataPreprocess()

    train_x, train_y = data_preprocess.return_train_data(list(range(10, 20)) + list(range(40, 62)))
    test_x, test_y = data_preprocess.return_test_data(list(range(10, 20)) + list(range(40, 62)))

    # linear_regression = LinearRegression(input_x_train=train_x, input_y_train=train_y, input_x_test=test_x, input_y_test=test_y)
    # linear_regression.train_normal_linear_regression()

    # test_x = test_x[:100]
    # test_y = test_y[:100]

    neural_network = NeuralNetwork(number_feature=train_x.shape[1], number_hidden=5, number_output=1
                                   , number_layer=5)
    neural_network.start_train(train_x.as_matrix(), train_y.as_matrix(), test_x=test_x.as_matrix(),
                               test_y=test_y.as_matrix(), number_iteration=30000)

    # random_forest_regression = RandomForestRegression(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    # random_forest_regression.start_train()
