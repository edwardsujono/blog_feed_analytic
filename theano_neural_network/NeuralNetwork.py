import theano
import theano.tensor as T
from theano_neural_network.utils import init_weights,init_bias
from sklearn.metrics import *
import numpy as np


class NeuralNetwork:

    def __init__(self, number_features, list_hidden_layer, learning_rate=0.001):

        theano.config.exception_verbosity = 'high'

        x = T.matrix('x')
        d = T.matrix('d')

        list_neurons = [number_features]
        list_neurons.extend(list_hidden_layer)
        list_neurons.extend([1])

        list_weights = []
        list_biases = []

        prev_output = x

        for i in range(1, len(list_neurons)):

            weight = init_weights(list_neurons[i - 1], list_neurons[i])
            bias = init_bias(list_neurons[i])

            list_weights.append(weight)
            list_biases.append(bias)

            if i < len(list_neurons)-1:
                prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)
            else:
                prev_output = T.dot(prev_output, weight) + bias

        cost = T.mean(T.sqr(d-prev_output))
        list_weights.extend(list_biases)
        params = list_weights

        grads = T.grad(cost, params)

        updates = [(param, param + learning_rate*param*grad) for param, grad in zip(params, grads)]

        self.train = theano.function(
            inputs=[x, d],
            updates=updates,
            outputs=[cost, prev_output],
            allow_input_downcast=True
        )

        self.test = theano.function(
            inputs=[x],
            outputs=[prev_output]
        )

    def start_train(self, train_x, train_y, epochs):
        print 'start_train'
        # train without mini batch
        train_trans_y = np.reshape(train_y, (len(train_y), 1))
        for i in range(epochs):

            cost, out = self.train(train_x, train_trans_y)
            if i % 500 == 0:
                print 'cost: %s, out: %s \n' % (cost, out)

    def start_test(self, test_x, test_y):

        cost, out = self.test(test_x)
        print 'cost: %s, out: %s \n' % (cost, r2_score(test_y, out))
