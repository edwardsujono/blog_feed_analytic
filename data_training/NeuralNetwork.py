import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score


class NN(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, number_layer=1):

        super(NN, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)

        self.hidden_array = []
        for i in range(number_layer):
            self.hidden_array.append(torch.nn.Linear(n_hidden, n_hidden))
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):

        fu = F.relu(self.hidden_1(x))

        fu_hidden_prev = F.relu(self.hidden_array[0](fu))

        fu_array = []
        for i in range(len(self.hidden_array)):
            fu_array.append(F.relu(self.hidden_array[i](fu_hidden_prev)))
            fu_hidden_prev = fu_array[i]

        y = self.output(fu_hidden_prev)
        return y


class NeuralNetwork:

    def __init__(self, number_feature, number_hidden, number_output, number_layer=1):

        self.net = NN(n_feature=number_feature, n_hidden=number_hidden, n_output=number_output, number_layer=number_layer)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss()

    def start_train(self, x, y, test_x, test_y, number_iteration=1000):

        x_input = torch.FloatTensor(x)
        y_input = torch.FloatTensor(y)

        buffer_y = y

        x = Variable(x_input)
        y = Variable(y_input)

        self.save_model(self.net)

        for i in range(1, number_iteration):
            prediction = self.net(x)
            loss = self.loss(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print ('iter: %d, loss: %s, r^2: %s \n' % (i, loss.data[0], r2_score(prediction.data.numpy(), buffer_y)))

        #search square error of it

        test_x = torch.FloatTensor(len(test_x), len(test_x[0]))
        test_y = torch.FloatTensor(len(test_y), 1)

        test_x = Variable(test_x)
        test_y = Variable(test_y)

        prediction_y = self.net(test_x)
        print ('loss: %s \n' % self.loss(prediction_y, test_y))

    def save_model(self, model):

        torch.save(model.state_dict(), "/Users/edwardsujono/Python_Project/blog_feed_analytic/model/model.pickle")
