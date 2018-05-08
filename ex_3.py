import time

import numpy as np
import math
import pickle


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Function(object):
    def calc(self, x):
        raise NotImplemented

    def dcalc(self, x, y, y_hat, weights, curr_layer):
        raise NotImplemented


class Softmax(Function):
    def calc(self, input):
        W, x, b = input
        eval = np.dot(x, W) + b
        softmax_vec = np.exp(eval) / np.sum(np.exp(eval))
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


class TwoLayeredNN(object):
    def __init__(self, layers_func, learning_rate, num_classes):
        self.eta = learning_rate
        self.num_classes = num_classes
        self.function_layers = layers_func["layers_func"]
        self.num_of_layers = len(self.function_layers)
        self.layer_weights_and_b_pairs = []
        for x_size, y_size in layers_func["sizes"]:
            self.layer_weights_and_b_pairs.append((np.random.rand(x_size, y_size), np.random.rand(1, y_size)))

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
        return zs, hs, hs[-1], train_example

    def __backprop_in_net__(self, forward_net_info):
        W2, b2 = self.layer_weights_and_b_pairs[1]
        zs, hs, y_hat, train = forward_net_info
        input, y = train

        dB_softmax = y_hat.copy()
        dB_softmax[y] -= 1

        dW_softmax = np.dot(hs[0], y_hat)
        dW_softmax[:, y] -= hs[0]

        dL_dH = np.dot(y_hat, W2) - W2[:, y]
        dH_dlayer1 = self.function_layers[0].dcalc(zs[0])
        dlayer1_dB1 = dL_dH * dH_dlayer1
        dlayer1_dW1 = dlayer1_dB1.copy()
        dlayer1_dW1 = np.outer(input, dlayer1_dW1)

        #W2,b2 = self.layer_weights_and_b_pairs[1]
        #Wx_b = zs[0].copy()
        #Wx_b[Wx_b > 0] = 1
        #Wx_b[Wx_b < 0] = 0
        #dW_relu = zs[0]
        #redce calculation
        #dB_relu = dW_relu.copy()
        #dW_relu[:, y] -= W2[:, y]
        #dW_relu = dW_relu * Wx_b * input


        #dB_relu[:, y] -= W2[:, y]
        #dB_relu = dB_relu * Wx_b

        return [dW_softmax, dlayer1_dW1], [dB_softmax, dlayer1_dB1]

    def learn(self, training_set, test_set, batch_size=1,  epocs=10, ):
        string_header = '|\t{0}\t|\t{1}\t|\t{2}\t|\t{3}\t|\t{4}\t|'
        string_underline = len(string_header) * '_'
        string_table_row = '|\t#{0}\t|\t{1}\t|\t{2}\t|\t{3}%\t|\t{4} seconds\t|'
        print(string_header.format('Epoc number', 'total loss', 'test set loss', 'accuracy', 'time'))
        for epoc_num in range(epocs):
            for batch in list(chunks(training_set, batch_size)):
                sum_weight_change, sum_bias_change = ([], [])
                for train_example in batch:
                    zs, hs, y_hat, _ = self.__forward_in_net__(train_example)
                    dW, dB = self.__backprop_in_net__((zs, hs, y_hat))
                    sum_weight_change.append(dW)
                    sum_bias_change.append(dB)
                self.__update__(sum_weight_change[0], sum_bias_change[0])
            time1 = time.time()
            loss, accuracy = self.compare_against_test_set(test_set)
            time2 = time.time()
            print(string_table_row.format(str(epoc_num), '0', loss, accuracy, (time2-time1)*1000.0))

    def __update__(self, dW, dB):
        W2, b2 = self.layer_weights_and_b_pairs[1]
        W1, b1 = self.layer_weights_and_b_pairs[1]
        W1 = W1 - self.eta * dW[0]
        b1 = b1 - self.eta * dB[0]
        W2 = W2 - self.eta * dW[1]
        b2 = b2 - self.eta * dB[1]

    def compare_against_test_set(self, test_set):
        num_right = 0
        loss = 0.0
        for test in test_set:
            input, y = test
            _, _, y_hat, _ = self.__forward_in_net__(input)
            prediction = np.argmax(y_hat)
            if prediction == y:
                num_right += 1
            loss += -np.log(y_hat[int(y)])
        return loss, num_right / len(test_set)


if __name__ == '__main__':
    print("Started loading data")
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    test_y = np.loadtxt("test.pred")
    print("Finished loading data")
    proper_data_set = zip(train_x, train_y)
    proper_test_set = zip(test_x, test_y)
    np.random.shuffle(proper_data_set)
    real_train_80, validation_20 = proper_data_set[:int(0.8 * len(train_x))], proper_data_set[int(0.8 * len(train_x)):]
    network = TwoLayeredNN(learning_rate=0.1,
                           num_classes=10,
                           layers_func={"layers_func": [Relu()],
                                        "sizes": [(784, 16), (16, 10)]})
    network.learn(real_train_80, test_set=proper_test_set)
