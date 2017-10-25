from preprocessing.DataPreprocess import DataPreprocess
from theano_neural_network.NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    data_preprocess = DataPreprocess()

    number_features = range(62)

    train_x, train_y = data_preprocess.return_train_data(number_features)
    test_x, test_y = data_preprocess.return_test_data(number_features)

    # this is solely intended for neural network analysis

    neural_network = NeuralNetwork(number_features=len(number_features), list_hidden_layer=[20, 20])
    neural_network.start_train(train_x.as_matrix(), train_y.as_matrix(), 1000)
