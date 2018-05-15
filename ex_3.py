import time
import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class PrettyPrinter(object):
    def __init__(self, seperator, hyper_parameters):
        self.hyper = hyper_parameters
        self.sep = seperator

    def __wrap__(self, str):
        print(self.sep * 100)
        print(str)
        print(self.sep * 100)

    def print_header(self):
        string_header = '|\t{0}\t|\t{1}\t|\t{2}\t|\t{3}\t|\t{4}\t|'
        learn_rate, hidden_layer_size, function, num_epocs, mini_batch_size, shuffle = self.hyper
        string_hyper = '\n\t'.join(['Hyper parameters are:', 'Learning rate: {0}', 'Hidden layer size: {1}', 'Function name: {2}', 'Number of epocs: {3}', 'Batch size: {4}', 'Shuffle every epoc: {5}'])
        self.__wrap__(string_hyper.format(learn_rate, hidden_layer_size, function.toString(), num_epocs, mini_batch_size, str(shuffle)))
        self.__wrap__(string_header.format('Epoc number', 'total loss', 'test set loss', 'accuracy', 'time'))

    def print_table_row(self, x):
        string_table_row = '|\t#{0}\t|\t{1}\t|\t{2}\t|\t{3}%\t|\t{4} seconds\t|'
        epoc, total_loss, test_set_loss, accuracy, time = x
        print(string_table_row.format(epoc, total_loss, test_set_loss, accuracy, time / 0.016667))


class Function(object):
    def calc(self, x):
        raise NotImplemented

    def dcalc(self, x, y, y_hat, weights, curr_layer):
        raise NotImplemented


class Softmax(Function):
    def calc(self, input):
        input -= input.max()
        softmax_vec = np.exp(input) / np.sum(np.exp(input))
        return softmax_vec

    def dcalc(self, x, y, y_hat, weights, curr_layer):
        dB = y_hat.copy()
        dB[y] -= 1

        dW = np.dot(x, y_hat)
        dW[:, y] -= x

        return dW, dB


class Relu():
    def calc(self, input):
        value = input
        value[value < 0] = 0
        return value

    def dcalc(self, hs):
        h_bigger_zero = hs.copy()
        h_bigger_zero[h_bigger_zero <= 0] = 0
        h_bigger_zero[h_bigger_zero > 0] = 1
        return h_bigger_zero

    def toString(self):
        return "Relu"


class TanH():
    def calc(self, input):
        return np.tanh(input)

    def dcalc(self, input):
        return 1 - np.square(np.tanh(input))

    def toString(self):
        return "TanH"


class Sigmoid():
    def calc(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def dcalc(self, input):
        return self.calc(input) * (1.0 - self.calc(input))

    def toString(self):
        return "Sigmoid"


def myRandom(size1, size2=None):
    t = 1 if size2 is None else size2
    eps = np.sqrt(6.0 / (size1 + t))
    return np.random.uniform(-eps, eps, (size1, size2)) if size2 is not None else np.random.uniform(-eps, eps, size1)


class TwoLayeredNN(object):
    def __init__(self, layers_func, learning_rate, printer=None):
        self.eta = learning_rate
        self.function_layers = layers_func["layers_func"]
        self.num_of_layers = len(self.function_layers)
        self.layer_weights_and_b_pairs = []
        self.printer = PrettyPrinter('*', (0, 0, "None", 0, 0)) if printer is None else printer
        for x_size, y_size in layers_func["sizes"]:
            self.layer_weights_and_b_pairs.append((myRandom(x_size, y_size), myRandom(y_size)))

    def __forward_in_net__(self, train_example):
        input, tag = train_example
        zs = []
        hs = []
        input_temp = input
        for function_activation, weight_and_b in zip(self.function_layers, self.layer_weights_and_b_pairs):
            W, b = weight_and_b
            z_i_temp = np.dot(input_temp, W) + b
            h_i_temp = function_activation.calc(z_i_temp)
            zs.append(z_i_temp)
            hs.append(h_i_temp)
            input_temp = hs[-1]
        return zs, hs, hs[-1], train_example

    def __backprop_in_net__(self, forward_net_info):
        W2, b2 = self.layer_weights_and_b_pairs[1]
        zs, hs, y_hat, train = forward_net_info
        input, y = train

        dB_softmax = y_hat.copy()
        dB_softmax[y] -= 1

        dW_softmax = np.outer(hs[0], y_hat)
        dW_softmax[:, y] -= hs[0]

        dL_dH = np.dot(W2, y_hat) - W2[:, y]
        dH_dlayer1 = self.function_layers[0].dcalc(zs[0])
        dlayer1_dB1 = dL_dH * dH_dlayer1
        dlayer1_dW1 = dlayer1_dB1.copy()
        dlayer1_dW1 = np.outer(input, dlayer1_dW1)
        return [dW_softmax, dlayer1_dW1], [dB_softmax, dlayer1_dB1]

    def learn(self, training_set, test_set, batch_size=1,  epocs=10):
        for epoc_num in range(1, epocs):
            np.random.shuffle(training_set)
            for batch in list(chunks(training_set, batch_size)):
                for train_example in batch:
                    zs, hs, y_hat, _ = self.__forward_in_net__(train_example)
                    dW, dB = self.__backprop_in_net__((zs, hs, y_hat, train_example))
                    self.__update__(dW, dB)

            time1 = time.time()
            loss, accuracy = self.compare_against_test_set(test_set)
            time2 = time.time()
            self.printer.print_table_row((str(epoc_num), 'Haim Homo', loss, accuracy * 100, time2 - time1))

    def __update__(self, dW, dB):
        W2, b2 = self.layer_weights_and_b_pairs[1]
        W1, b1 = self.layer_weights_and_b_pairs[0]
        W1 = W1 - self.eta * dW[1]
        b1 = b1 - self.eta * dB[1]
        W2 = W2 - self.eta * dW[0]
        b2 = b2 - self.eta * dB[0]
        self.layer_weights_and_b_pairs = [(W1, b1), (W2, b2)]

    def learn2(self, training_set, test_set, batch_size=1, epocs=10):
        for epoc_num in range(1, epocs):
            np.random.shuffle(training_set)
            for batch in list(chunks(training_set, batch_size)):
                for train_example in batch:
                    zs, hs, y_hat, _ = self.__forward_in_net__(train_example)
                    dW, dB = self.__backprop_in_net__((zs, hs, y_hat, train_example))
                    self.__update2__(dW, dB, epoc_num)

            time1 = time.time()
            loss, accuracy = self.compare_against_test_set(test_set)
            time2 = time.time()
            self.printer.print_table_row((str(epoc_num), 'Haim Homo', loss, accuracy * 100, time2 - time1))

    def __update2__(self, dW, dB, n):
        W2, b2 = self.layer_weights_and_b_pairs[1]
        W1, b1 = self.layer_weights_and_b_pairs[0]
        W1 = W1 - self.eta[n - 1] * dW[1]
        b1 = b1 - self.eta[n - 1] * dB[1]
        W2 = W2 - self.eta[n - 1] * dW[0]
        b2 = b2 - self.eta[n - 1] * dB[0]
        self.layer_weights_and_b_pairs = [(W1, b1), (W2, b2)]

    def compare_against_test_set(self, test_set):
        num_right = 0
        loss = []
        for test in test_set:
            input, y = test
            _, _, y_hat, _ = self.__forward_in_net__(test)
            prediction = np.argmax(y_hat)
            if prediction == y:
                num_right += 1
            loss.append(-np.log(y_hat[y]))
        loss_np = np.array(loss)
        return loss_np.mean(), float(num_right) / len(test_set)

    def predict(self, input):
        _, _, y_hat, _ = self.__forward_in_net__((input, -1))
        return np.argmax(y_hat)


if __name__ == '__main__':
    learn_rate, hidden_layer_size, function, num_epocs, mini_batch_size, shuffle_every_epoc = 0.005, 256, Relu(), 12, 1, True
    learn_rate = (5 * [0.015]) + (5*[0.01]) + (2 * [0.0015])
    printer = PrettyPrinter('*', (learn_rate, hidden_layer_size, function, num_epocs, mini_batch_size, shuffle_every_epoc))
    print("Started loading data")
    train_x = np.loadtxt("train_x") / 255.0
    train_y = np.loadtxt("train_y", dtype=np.int)
    test_x = np.loadtxt("test_x") / 255.0
    print("Finished loading data")
    proper_data_set = zip(train_x, train_y)
    #proper_test_set = zip(test_x, test_y)
    np.random.shuffle(proper_data_set)
    real_train_80, validation_20 = proper_data_set[:int(0.8 * len(proper_data_set))], proper_data_set[int(0.8 * len(proper_data_set)):]
    network = TwoLayeredNN(learning_rate=learn_rate,
                           printer=printer,
                           layers_func={"layers_func": [function, Softmax()],
                                        "sizes": [(784, hidden_layer_size), (hidden_layer_size, 10)]})
    printer.print_header()
    network.learn2(real_train_80, test_set=validation_20, batch_size=mini_batch_size, epocs=num_epocs)
