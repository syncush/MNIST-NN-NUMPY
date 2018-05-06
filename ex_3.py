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


class Relu(Function):
    def calc(self, input):
        value = input
        value[value < 0] = 0
        return value

    def dcalc(self, x, y, y_hat, weights, curr_layer, input):
        h_bigger_zero = np.dot()
        h_bigger_zero[h_bigger_zero <= 0] = 0
        h_bigger_zero[h_bigger_zero > 0] = 1

        temp_d = np.dot(weights[curr_layer + 1][:, y], y_hat)
        temp_d -= weights[curr_layer + 1][:, y]
        temp_d *= h_bigger_zero
        temp_d np.dot(x, temp_d)


class TwoLayeredNN(object):
    def __init__(self, layers_func, learning_rate, num_classes, num_epocs):
        self.eta = learning_rate
        self.epocs = num_epocs
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
        zs, hs, y_hat, train = forward_net_info
        input, y = train

        dB_softmax = y_hat.copy()
        dB_softmax[y] -= 1

        dW_softmax = np.dot(hs[0], y_hat)
        dW_softmax[:, y] -= hs[0]

        W2,b2 = self.layer_weights_and_b_pairs[1]
        Wx_b = z[0].copy()
        Wx_b[Wx_b > 0] = 1
        Wx_b[Wx_b < 0] = 0
        dW_relu = np.dot(W2, y_hat)
        #redce calculation
        dB_relu = dW_relu.copy()
        dW_relu[:, y] -= W2[:, y]
        dW_relu = dW_relu * Wx_b * input


        dB_relu[:, y] -= W2[:, y]
        dB_relu = dB_relu * Wx_b



        return [dW_softmax, dW_relu], [dB_softmax, dB_relu]

    def learn(self, training_set, batch_size=1):
        for _ in self.epocs:
            for batch in list(chunks(training_set, batch_size)):
                sum_weight_change, sum_bias_change = ([], [])
                for train_example in batch:
                    zs, hs, model_output = self.__forward_in_net__(train_example)
                    dW, dB = self.__backprop_in_net__((zs, hs, model_output))
                    sum_weight_change.append(dW)
                    sum_bias_change.append(dB)

    def compare_against_test_set(self, test_set):
        num_right = 0
        for test in test_set:
            input, tag = test
            _, _, model_output = self.__forward_in_net__(input)
            prediction = np.argmax(model_output) + 1
            if prediction == tag:
                num_right += 1
        return num_right / len(test_set)


if __name__ == '__main__':
    train_x, train_y, test_x = pickle.loads(open("./pickled.bin", "rb").read())
    function_sigmoid = Sigmoid()
    proper_data_set = zip(train_x, train_y)
    np.random.shuffle(proper_data_set)
    real_train_80, validation_20 = proper_data_set[:int(0.8 * len(train_x))], proper_data_set[int(0.8 * len(train_x)):]
    network = TwoLayeredNN(learning_rate=0.1,
                           num_classes=10,
                           layers_func={"layers_func": [function_sigmoid, function_sigmoid],
                                        "sizes": [(784, 16), (16, 10)]})
