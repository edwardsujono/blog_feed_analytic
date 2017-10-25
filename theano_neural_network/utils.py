import numpy as np
import theano


def init_weights(n_in, n_out):
    weight = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_in + n_out)),
            high=4 * np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),
        dtype=theano.config.floatX)
    return theano.shared(value=weight, borrow=True, name='w')


def init_bias(n):
    return theano.shared(value=np.zeros(n, dtype=theano.config.floatX), borrow=True, name='b')